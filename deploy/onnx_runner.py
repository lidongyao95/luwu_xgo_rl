#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 真机部署 —— 用 luwu_mjlab 训出来的 Mini2S-Walk 策略跑推理

策略接口（从 policy.onnx 元数据读出，已核对）：
    obs [1,56] → actions [1,15]
    obs  = base_ang_vel(3), projected_gravity(3), command(3),
           phase(2), joint_pos(15), joint_vel(15), actions(15)
    action_scale = 0.25
    default_joint_pos = 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,-1.570,1.350   (弧度)
    狗端目标频率 = 40 Hz。姿态与关节差分用各自的实际串口采样间隔。

用法：
    python3 onnx_runner.py --dry             # 只算不发动，打印观测/动作
    python3 onnx_runner.py --probe 31 3      # 单独点动右后腿的靠脚关节
    python3 onnx_runner.py --run             # 闭环跑策略（⚠️ 会让狗动）

已知的两个妥协（必须清楚）：
  1. 这台狗的固件（M-7.0.0b8）上 dog.read_imu() 返回的是死数据（每次完全相同，
     含 2.99e36 这种垃圾值），只有 read_roll/pitch/yaw 是活的。
     所以 base_ang_vel 只能用姿态角差分 + 低通估计，噪声偏大。
  2. 手臂三个关节在真机上收在 (-65, 50, -75)°，和仿真默认 (0, -90, 77)° 完全不同。
     手臂总质量只有 ~40 g，对动力学影响可忽略，所以这里**不驱动手臂**，
     观测里也把手臂按仿真默认值（相对量 0）上报，避免给策略喂分布外的输入。
"""

import argparse
import math
import os
import signal
import struct
import time

import numpy as np
import onnxruntime as ort

# 策略文件放在 deploy/models/ 下；找不到再回退到狗上的旧部署路径。
POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "models", "mini2s_walk_policy.onnx")
if not os.path.exists(POLICY):
    POLICY = "/home/pi/Chapter4/mini2s_walk_policy.onnx"
CTRL_HZ = 40.0

# ONNX 元数据里的关节顺序（15 个）
JOINT_NAMES = [
    "fl_hip", "fl_thigh", "fl_calf",
    "fr_hip", "fr_thigh", "fr_calf",
    "br_hip", "br_thigh", "br_calf",
    "bl_hip", "bl_thigh", "bl_calf",
    "arm_yaw", "arm_thigh", "arm_calf",
]
DEFAULT_POS = np.deg2rad([0, 57.3, 0, 0, 57.3, 0, 0, 57.3, 0, 0, 57.3, 0,
                          0, -90.0, 77.4])
ACTION_SCALE = 0.25

# Mini2S 以靠近机身的 hip、中间的 thigh、靠脚的 calf 排列策略输出。
# SDK 的角度反馈在每条腿内按相反的顺序返回。
LEG_SERVO_IDS = [13, 12, 11, 23, 22, 21, 33, 32, 31, 43, 42, 41]
LEG_READ_INDICES = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9]
# MINI 舵机编码范围与脚本原有 ±90° 上限的交集。
LEG_TARGET_LIMITS = np.deg2rad([(-31, 31), (-66, 90), (-73, 57)] * 4)

TILT_ABORT_DEG = 25.0
ANG_VEL_ALPHA = 0.12      # 姿态差分估计角速度的低通系数，越小越平滑
SLEW_DEG = 4.0            # 每个控制周期允许的最大关节变化（度）→ 4°×40Hz = 160°/s
# 舵机写入后舵机板偶尔会延迟应答；单帧读失败不应终止试跑。
READ_ATTEMPTS = 3
READ_TIMEOUT = 0.03
CMD = {"vx": 0.0, "wz": 0.0, "t": 0.0}   # 策略单位：m/s, rad/s
CMD_TIMEOUT = 0.5                        # 超时回零，和键盘失联保护一致
PHASE_PERIOD = 0.4
CMD_DEADBAND = 0.1


# ----------------------------------------------------------- 观测组装 --

class Imu:
    """read_imu() 在这台狗上是坏的，只能用姿态角 + 差分。"""

    def __init__(self, dog, alpha=ANG_VEL_ALPHA):
        self.dog = dog
        self.alpha = alpha
        self.prev = None
        self.prev_t = None
        self.omega = np.zeros(3, dtype=np.float32)

    def read_attitude(self):
        """返回 (roll, pitch, yaw) 弧度。"""
        r = math.radians(self.dog.read_roll())
        p = math.radians(self.dog.read_pitch())
        y = math.radians(self.dog.read_yaw())
        return r, p, y

    def update(self):
        r, p, y = self.read_attitude()
        now = time.monotonic()
        dt = 0.0 if self.prev_t is None else now - self.prev_t
        if self.prev is not None and dt > 0:
            dr = _wrap(r - self.prev[0]) / dt
            dp = _wrap(p - self.prev[1]) / dt
            dy = _wrap(y - self.prev[2]) / dt
            # 欧拉角速度 → 机体角速度（ZYX 欧拉）
            sr, cr = math.sin(r), math.cos(r)
            sp, cp = math.sin(p), math.cos(p)
            wx = dr - dy * sp
            wy = dp * cr + dy * sr * cp
            wz = -dp * sr + dy * cr * cp
            raw = np.array([wx, wy, wz], dtype=np.float32)
            alpha = 1.0 - (1.0 - self.alpha) ** (dt * CTRL_HZ)
            self.omega = alpha * raw + (1.0 - alpha) * self.omega
        self.prev = (r, p, y)
        self.prev_t = now
        return r, p, y

    @staticmethod
    def projected_gravity(r, p):
        """机体系下的重力方向 R^T·[0,0,-1]（ZYX 欧拉）。"""
        return np.array([math.sin(p),
                         -math.sin(r) * math.cos(p),
                         -math.cos(r) * math.cos(p)], dtype=np.float32)


def _wrap(a):
    """把角度差折到 [-pi, pi]。用取模，避免遇到异常大值时死循环。"""
    if not math.isfinite(a):
        return 0.0
    return (a + math.pi) % (2 * math.pi) - math.pi


class PolicyRunner:
    def __init__(self, policy_path=POLICY, verbose=True):
        self.sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        inp, out = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        if inp.shape != [1, 56] or out.shape != [1, 15]:
            raise ValueError(f"Mini2S ONNX dimensions differ: {inp.shape} -> {out.shape}")
        self.inp, self.out = inp.name, out.name
        meta = self.sess.get_modelmeta().custom_metadata_map
        names = meta.get("joint_names", "").split(",")
        if names != [name + "_joint" for name in JOINT_NAMES]:
            raise ValueError(f"Mini2S ONNX joint order differs: {names}")
        if float(meta["action_scale"]) != ACTION_SCALE:
            raise ValueError("Mini2S ONNX action scale differs")
        self.action = np.zeros(15, dtype=np.float32)
        self.t0 = time.time()

        if verbose:
            print("[runner] 策略加载完成")
            print("[runner] 关节顺序:", ",".join(names))

    def reset_clock(self):
        self.t0 = time.time()

    def phase(self, t, cmd):
        frac = (t % PHASE_PERIOD) / PHASE_PERIOD
        s, c = math.sin(frac * 2 * math.pi), math.cos(frac * 2 * math.pi)
        if float(np.linalg.norm(cmd)) < CMD_DEADBAND:
            return np.zeros(2, dtype=np.float32)
        return np.array([s, c], dtype=np.float32)

    def infer(self, obs):
        self.action = self.sess.run(None, {self.inp: obs[None, :]})[0][0].astype(np.float32)
        return self.action


def build_obs(imu, joints_rad, joint_vel, cmd, policy, t, action_prev):
    """按 ONNX 元数据显示的顺序拼 56 维。"""
    r, p, _ = imu.prev
    parts = [
        imu.omega,                              # base_ang_vel 3
        Imu.projected_gravity(r, p),            # projected_gravity 3
        np.asarray(cmd, dtype=np.float32),      # command 3
        policy.phase(t, cmd),                   # phase 2
        joints_rad - DEFAULT_POS,               # joint_pos 15 (相对)
        joint_vel,                              # joint_vel 15
        action_prev,                            # actions 15
    ]
    obs = np.concatenate(parts).astype(np.float32)
    # 串口偶尔读失败会带出 NaN/Inf，先消毒再喂网络
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- 主流程 --

class Robot:
    def __init__(self, read_only=False):
        import serial
        from xgolib import XGO_DOG

        # XGO(auto) 会执行 reset 和舵机初始化。这里只建立 UART 和解析器。
        self._sdk = XGO_DOG.__init__.__globals__
        self._sdk["changePara"]("xgomini")
        self.dog = object.__new__(XGO_DOG)
        self.dog.ser = serial.Serial("/dev/ttyAMA0", 115200, timeout=0.05,
                                     write_timeout=0.05, exclusive=True)
        self.dog.verbose = False
        self.dog.rx_FLAG = self.dog.rx_COUNT = self.dog.rx_ADDR = self.dog.rx_LEN = 0
        self.dog.rx_data = bytearray(50)
        self.read_only = read_only
        self.read_retries = 0
        self.read_failures = 0
        try:
            fw = self._read_register("FIRMWARE_VERSION", 10, (10,)).decode("ascii").strip("\0")
            if not fw.startswith("M-"):
                raise RuntimeError(f"Unexpected Mini firmware: {fw!r}")
            battery = self._read_register("BATTERY", 1, (1,))[0]
            if battery > 100 or (not read_only and battery < 25):
                raise RuntimeError(f"Invalid or low battery: {battery}")
            print(f"[robot] 固件 {fw}, 电量 {battery}%")
        except BaseException:
            self.close()
            raise

    def _read_register(self, name, requested, sizes, attempts=READ_ATTEMPTS):
        """读寄存器；舵机写入后偶发丢帧时重发，连续失败才报错。"""
        address = self._sdk["XGOorder"][name][0]
        for attempt in range(attempts):
            self.dog.rx_FLAG = 0
            self.dog.rx_COUNT = 0
            self.dog._XGO_DOG__read(address, requested)
            if (self.dog._XGO_DOG__unpack(timeout=READ_TIMEOUT)
                    and self.dog.rx_ADDR == address and self.dog.rx_TYPE == 0x12
                    and self.dog.rx_COUNT + 1 in sizes):
                if attempt:
                    self.read_retries += 1
                return bytes(self.dog.rx_data[:self.dog.rx_COUNT + 1])
        self.read_failures += 1
        raise RuntimeError(f"Invalid UART feedback for {name}")

    def _read_attitude(self, name):
        value, = struct.unpack("<f", self._read_register(name, 4, (4,)))
        if not math.isfinite(value) or abs(value) > 360.0:
            raise RuntimeError(f"Invalid attitude angle for {name}: {value}")
        return round(value, 2)

    def read_roll(self):
        return self._read_attitude("ROLL")

    def read_pitch(self):
        return self._read_attitude("PITCH")

    def read_yaw(self):
        return self._read_attitude("YAW")

    def read_joints(self):
        """按 ONNX 顺序返回关节角，单位弧度。"""
        raw = self._read_register("MOTOR_ANGLE", 15, (15, 16))
        limits = self._sdk["XGOparam"]["MOTOR_LIMIT"]
        q = DEFAULT_POS.copy()
        degrees = [self._sdk["conver2float"](raw[i], limits[i % 3]) for i in LEG_READ_INDICES]
        q[:12] = np.deg2rad(degrees)
        return q

    def send_legs(self, q_rad):
        """只下发 12 个腿部舵机目标。"""
        if self.read_only:
            raise RuntimeError("Actuator write in read-only mode")
        degs = np.rad2deg(q_rad[:12])
        self.dog.motor(LEG_SERVO_IDS, [float(v) for v in degs])

    def send_joint(self, servo_id, radians):
        if self.read_only:
            raise RuntimeError("Actuator write in read-only mode")
        self.dog.motor(servo_id, math.degrees(radians))

    def stop(self):
        if not self.read_only:
            self.dog.stop()

    def close(self):
        if getattr(self, "read_retries", 0) or getattr(self, "read_failures", 0):
            print(f"[robot] UART 重试 {self.read_retries} 次，失败 {self.read_failures} 次",
                  flush=True)
            self.read_retries = self.read_failures = 0
        if self.dog.ser.is_open:
            self.dog.ser.close()


class JointFeedback:
    def __init__(self, robot):
        self.robot = robot
        self.prev = None
        self.prev_t = None

    def read(self):
        q = self.robot.read_joints()
        now = time.monotonic()
        jvel = np.zeros(15, dtype=np.float32)
        if self.prev_t is not None and now > self.prev_t:
            jvel[:12] = (q[:12] - self.prev[:12]) / (now - self.prev_t)
        self.prev, self.prev_t = q, now
        return q, jvel


def run_dry(robot, policy, seconds=3.0):
    imu = Imu(robot)
    feedback = JointFeedback(robot)
    dt = 1.0 / CTRL_HZ
    feedback.read()
    imu.update()
    policy.reset_clock()
    t_start = time.monotonic()
    n = 0
    lat = []
    while time.monotonic() - t_start < seconds:
        loop0 = time.monotonic()
        imu.update()
        q, jvel = feedback.read()

        cmd = np.zeros(3, dtype=np.float32)
        obs = build_obs(imu, q, jvel, cmd, policy, time.monotonic() - t_start, policy.action)
        t_inf = time.monotonic()
        act = policy.infer(obs)
        if not np.isfinite(act).all():
            raise RuntimeError("Nonfinite policy action")
        lat.append((time.monotonic() - t_inf) * 1000)

        q_target = DEFAULT_POS + ACTION_SCALE * act
        n += 1
        if n % 10 == 0:
            print("[dry] t=%5.2fs obs[:9]=%s" % (time.monotonic() - t_start,
                                                 np.round(obs[:9], 3)))
            print("       act[:6]=%s  目标(度)=%s" % (
                np.round(act[:6], 3), np.round(np.rad2deg(q_target[:6]), 1)))
        time.sleep(max(0.0, dt - (time.monotonic() - loop0)))

    print("\n[dry] 共 %d 帧, 实际 %.1f Hz, ONNX 平均 %.2f ms"
          % (n, n / (time.monotonic() - t_start), float(np.mean(lat))))
    print("[dry] 未向舵机下发任何指令")


def run_probe_group(robot, first_id, deg, hold=1.8):
    """整条腿（hip/thigh/calf）一起转，动作更明显。"""
    q = robot.read_joints()
    if first_id not in (11, 21, 31, 41):
        print("支持的腿起始舵机号: 11 / 21 / 31 / 41")
        return
    i0 = LEG_SERVO_IDS.index(first_id) // 3 * 3
    print("[probe-group] 舵机组 %d-%d 抖动 %+.1f°（保持 %.1fs）"
          % (first_id, first_id + 2, deg, hold))
    target = q.copy()
    for k in range(3):
        target[i0 + k] = q[i0 + k] + math.radians(deg if k != 2 else -deg)
    robot.send_legs(target)
    try:
        time.sleep(hold)
    finally:
        robot.send_legs(q)
        print("[probe-group] 已复位")


def run_probe(robot, servo_id, deg, hold=1.5):
    """把单个舵机转 deg 度、保持 hold 秒、然后自动复位。

    用来核对「MJCF 关节顺序」和「XGO 舵机编号」的对应关系：
    看哪条腿动了，就知道 31-33 到底是左后还是右后。
    """
    q = robot.read_joints()
    if servo_id not in LEG_SERVO_IDS:
        print("支持的舵机号:", LEG_SERVO_IDS)
        return
    idx = LEG_SERVO_IDS.index(servo_id)
    before = float(np.rad2deg(q[idx]))
    print("[probe] %d 号舵机 当前 %.1f° → %.1f°（保持 %.1fs 后自动复位）"
          % (servo_id, before, before + deg, hold))

    if not math.isfinite(deg) or abs(deg) > 5:
        raise ValueError("Probe offset must be within ±5°")
    target = float(np.clip(q[idx] + math.radians(deg), *LEG_TARGET_LIMITS[idx]))
    try:
        robot.send_joint(servo_id, target)
        time.sleep(hold)
    finally:
        robot.send_joint(servo_id, q[idx])
        print("[probe] 已复位")


def start_cmd_server(port):
    """开一个小 HTTP 接口，让本机键盘遥控驱动策略。"""
    from flask import Flask, jsonify, request

    app = Flask("cmdsrv")

    @app.route("/cmd")
    def _cmd():
        CMD["vx"] = max(-0.30, min(0.35, float(request.args.get("vx", 0))))
        CMD["wz"] = max(-0.45, min(0.45, float(request.args.get("vyaw", 0))))
        CMD["t"] = time.time()
        return jsonify(CMD)

    @app.route("/stop")
    def _stop():
        CMD.update(vx=0.0, wz=0.0, t=0.0)
        return jsonify({"stopped": True})

    print("[cmdsrv] 指令接口 http://0.0.0.0:%d/cmd?vx=..&vyaw=.." % port)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


def current_cmd():
    if time.time() - CMD["t"] > CMD_TIMEOUT:
        return np.zeros(3, dtype=np.float32)
    return np.array([CMD["vx"], 0.0, CMD["wz"]], dtype=np.float32)


def run_closed_loop(robot, policy, seconds=10.0):
    """⚠️ 会让狗真的动。指令来自 /cmd，超时自动回零。"""
    imu = Imu(robot)
    feedback = JointFeedback(robot)
    dt = 1.0 / CTRL_HZ
    policy.reset_clock()

    print("[run] 3 秒后开始，先保持零指令…… Ctrl+C 停")
    time.sleep(3.0)

    # 从当前实际姿态起步，避免第一帧就跳变
    q, _ = feedback.read()
    q_cmd = q.copy()
    imu.update()
    max_step = math.radians(SLEW_DEG)

    t_start = time.monotonic()
    n = 0
    try:
        while time.monotonic() - t_start < seconds:
            loop0 = time.monotonic()
            imu.update()
            r, p, _ = imu.prev
            if abs(math.degrees(r)) > TILT_ABORT_DEG or abs(math.degrees(p)) > TILT_ABORT_DEG:
                print("[run] 倾角超限 roll=%.1f° pitch=%.1f°，停止"
                      % (math.degrees(r), math.degrees(p)))
                break
            q, jvel = feedback.read()

            cmd = current_cmd()
            obs = build_obs(imu, q, jvel, cmd, policy, time.monotonic() - t_start, policy.action)
            act = policy.infer(obs)

            if not np.isfinite(act).all():
                raise RuntimeError("Nonfinite policy action")
            q_target = DEFAULT_POS + ACTION_SCALE * act
            q_target[:12] = np.clip(q_target[:12], LEG_TARGET_LIMITS[:, 0], LEG_TARGET_LIMITS[:, 1])
            # 速率限制：每周期最多变 SLEW_DEG，防止猛甩
            q_target = np.clip(q_target, q_cmd - max_step, q_cmd + max_step)
            q_cmd = q_target
            robot.send_legs(q_target)

            n += 1
            if n % 20 == 0:
                print("[run] t=%4.1fs roll=%+5.1f° pitch=%+5.1f° cmd=[%.2f %.2f]"
                      % (time.monotonic() - t_start, math.degrees(r), math.degrees(p),
                         cmd[0], cmd[2]))
            time.sleep(max(0.0, dt - (time.monotonic() - loop0)))
    finally:
        robot.stop()
        print("[run] 停止，共 %d 帧" % n)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true", help="只计算不发动")
    mode.add_argument("--run", action="store_true", help="闭环跑策略（会动）")
    mode.add_argument("--probe", nargs=2, metavar=("SERVO_ID", "DEG"),
                    help="单独转一个舵机，用于核对关节映射")
    mode.add_argument("--probe-group", nargs=2, metavar=("FIRST_SERVO_ID", "DEG"),
                    help="整条腿三个关节一起转，更容易看清是哪条腿")
    ap.add_argument("--seconds", type=float, default=5.0, help="运行时长，默认 5 秒")
    ap.add_argument("--serve-cmd", type=int, default=0,
                    help="在该端口开 /cmd 指令接口，供本机键盘遥控")
    ap.add_argument("--policy", default=POLICY)
    args = ap.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        ap.error("--seconds must be finite and positive")

    if not os.path.exists(args.policy):
        raise SystemExit("找不到策略文件: " + args.policy)

    policy = PolicyRunner(args.policy)
    robot = Robot(read_only=args.dry)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)

    try:
        if args.dry:
            run_dry(robot, policy, args.seconds)
        elif args.probe:
            run_probe(robot, int(args.probe[0]), float(args.probe[1]))
        elif args.probe_group:
            run_probe_group(robot, int(args.probe_group[0]), float(args.probe_group[1]))
        elif args.run:
            if args.serve_cmd:
                import threading
                threading.Thread(target=start_cmd_server, args=(args.serve_cmd,),
                                 daemon=True).start()
                time.sleep(1.0)
            run_closed_loop(robot, policy, args.seconds)
    except KeyboardInterrupt:
        print("\n[main] 中断")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
