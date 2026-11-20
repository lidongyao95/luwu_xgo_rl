#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在本机（Mac）上跑 ONNX 策略，实时控制真狗

架构：
    Mac                                             狗 CM5
    ┌──────────────────────────────┐              ┌─────────────────────┐
    │ 40 Hz 循环                    │  GET /step   │ dog_bridge.py       │
    │  观测拼装 → ONNX 推理 → 动作   │ ───────────> │  下发 12 路角度      │
    │                              │ <─────────── │  回读 15 关节 + IMU  │
    └──────────────────────────────┘              └─────────────────────┘

每个控制周期只走**一次**网络往返（/step 先写后读），把 WiFi 延迟压到最小。

用法：
    python3 mac_runner.py --dry                 # 只算不发动
    python3 mac_runner.py --run --seconds 60    # 真跑
    python3 mac_runner.py --run --cmd            # 真跑并开键盘遥控（8081）

⚠️ WiFi 是 10–120 ms 抖动，40 Hz 的周期只有 25 ms。跑起来会看到实际频率明显低于 40 Hz，
   控制周期抖动也大。如果策略表现很差，优先怀疑这一点，而不是策略本身。
"""

import argparse
import math
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_runner import (ACTION_SCALE, CTRL_HZ, DEFAULT_POS, POLICY,
                         TILT_ABORT_DEG, PolicyRunner, build_obs)

CMD = {"vx": 0.0, "wz": 0.0, "t": 0.0}
CMD_TIMEOUT = 0.5


class RemoteImu:
    """从狗端回读的 roll/pitch/yaw 差分出角速度。"""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.prev = (0.0, 0.0, 0.0)
        self.omega = np.zeros(3, dtype=np.float32)
        self._first = True

    def update(self, roll, pitch, yaw, dt):
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
        if not self._first and dt > 1e-3:
            dr = _wrap(r - self.prev[0]) / dt
            dp = _wrap(p - self.prev[1]) / dt
            dy = _wrap(y - self.prev[2]) / dt
            sr, cr = math.sin(r), math.cos(r)
            sp, cp = math.sin(p), math.cos(p)
            raw = np.array([dr - dy * sp,
                            dp * cr + dy * sr * cp,
                            -dp * sr + dy * cr * cp], dtype=np.float32)
            self.omega = self.alpha * raw + (1 - self.alpha) * self.omega
        self.prev = (r, p, y)
        self._first = False
        return r, p, y


def _wrap(a):
    if not math.isfinite(a):
        return 0.0
    return (a + math.pi) % (2 * math.pi) - math.pi


class RemoteRobot:
    def __init__(self, host, port):
        self.base = "http://%s:%d" % (host, port)
        self.sess = requests.Session()
        self.last_q = None
        self.rtt_ms = []

    def state(self):
        r = self.sess.get(self.base + "/state", timeout=2.0)
        d = r.json()
        self.last_q = np.deg2rad(np.asarray(d["q"][:15], dtype=np.float64))
        self.last_q[12:15] = DEFAULT_POS[12:15]      # 手臂按仿真默认上报
        return d

    def step(self, q_deg12):
        """下发 12 个腿关节角并回读状态（一次往返）。"""
        qs = ",".join("%.2f" % v for v in q_deg12)
        t0 = time.time()
        r = self.sess.get(self.base + "/step", params={"q": qs}, timeout=1.5)
        self.rtt_ms.append((time.time() - t0) * 1000)
        d = r.json()
        q = np.deg2rad(np.asarray(d["q"][:15], dtype=np.float64))
        q[12:15] = DEFAULT_POS[12:15]
        self.last_q = q
        return d, q

    def stop(self):
        try:
            self.sess.get(self.base + "/stop", timeout=1.0)
        except Exception:
            pass

    def rtt_stats(self):
        if not self.rtt_ms:
            return (0.0, 0.0)
        a = np.asarray(self.rtt_ms)
        return float(a.mean()), float(np.percentile(a, 95))


def current_cmd():
    if time.time() - CMD["t"] > CMD_TIMEOUT:
        return np.zeros(3, dtype=np.float32)
    return np.array([CMD["vx"], 0.0, CMD["wz"]], dtype=np.float32)


def start_cmd_server(port):
    from flask import Flask, jsonify, request
    a = Flask("mac_cmd")

    @a.route("/cmd")
    def _c():
        CMD["vx"] = max(-0.8, min(0.8, float(request.args.get("vx", 0))))
        CMD["wz"] = max(-1.5, min(1.5, float(request.args.get("vyaw", 0))))
        CMD["t"] = time.time()
        return jsonify(CMD)

    @a.route("/stop")
    def _s():
        CMD.update(vx=0.0, wz=0.0, t=0.0)
        return jsonify({"stopped": True})

    print("[cmd] 指令接口 http://127.0.0.1:%d/cmd" % port)
    a.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


KEY_TIMEOUT = 0.30
KEY_MODE = {"on": False}
LOOP_ALIVE = {"ok": False, "err": ""}
MAX_FAILS = 3
VX_KEY, VYAW_KEY = 0.35, 0.8


def keyboard_control():
    keyboard_control.last_line = ""
    """在主线程里直接改 CMD，不走 HTTP —— 少一层、少几十毫秒。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    held = {}
    print("\n键盘: W/A/S/D 驱动 · 空格停 · Q 退出（直接改内存里的指令，无网络开销）\n")
    try:
        tty.setcbreak(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.01)
            now = time.time()
            if r:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    break
                if ch == " ":
                    held.clear()
                    CMD.update(vx=0.0, wz=0.0, t=0.0)
                    print("\r停车                          ", end="", flush=True)
                    continue
                if ch == "\x1b":
                    select.select([sys.stdin], [], [], 0.01)
                    third = sys.stdin.read(1) if sys.stdin.readable() else ""
                    ch = {"A": "w", "B": "s", "C": "d", "D": "a"}.get(third, "")
                ch = ch.lower()
                if ch in "wasd":
                    held[ch] = now
            if not LOOP_ALIVE["ok"]:
                print("\n[keys] 策略循环已停止（%s），遥控退出"
                      % (LOOP_ALIVE["err"] or "正常结束"))
                break

            for k in [k for k, t in held.items() if now - t > KEY_TIMEOUT]:
                del held[k]

            vx = (VX_KEY if "w" in held else 0.0) - (VX_KEY if "s" in held else 0.0)
            wz = (VYAW_KEY if "a" in held else 0.0) - (VYAW_KEY if "d" in held else 0.0)
            CMD.update(vx=vx, wz=wz, t=now)

            # 只在状态变化时重绘，否则终端会被刷爆
            line = "  %s  vx=%+.2f  wz=%+.2f " % ("".join(sorted(held)) or "·", vx, wz)
            if line != keyboard_control.last_line:
                keyboard_control.last_line = line
                print("\r" + line, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        CMD.update(vx=0.0, wz=0.0, t=0.0)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\n[keys] 已退出，停车")


def loop(robot, policy, seconds, dry=False):
    imu = RemoteImu()
    dt_nom = 1.0 / CTRL_HZ

    d0 = robot.state()
    imu.update(d0["roll"], d0["pitch"], d0["yaw"], dt_nom)
    q_prev = robot.last_q.copy()
    policy.reset_clock()

    LOOP_ALIVE["ok"] = True
    print("[loop] 3 秒后开始，Ctrl+C 停")
    time.sleep(3)

    t_start = time.time()
    t_prev = time.time()
    n = 0
    fails = 0
    periods = []
    try:
        while time.time() - t_start < seconds:
            loop0 = time.time()
            dt = max(1e-3, loop0 - t_prev)
            t_prev = loop0

            q_now = robot.last_q.copy() if robot.last_q is not None else q_prev
            jvel = np.zeros(15, dtype=np.float32)
            jvel[:12] = ((q_now - q_prev)[:12] / dt).astype(np.float32)

            cmd = current_cmd()
            obs = build_obs(imu, q_now, jvel, cmd, policy,
                            time.time() - t_start, policy.action)
            act = policy.infer(obs)

            if not np.isfinite(act).all():
                print("[loop] 动作 NaN，跳过")
                continue

            q_target = np.clip(DEFAULT_POS + ACTION_SCALE * act,
                               -math.pi / 2, math.pi / 2)
            degs = np.rad2deg(q_target[:12])

            if dry:
                if n % 10 == 0:
                    print("[dry] t=%4.1f q=%s" % (time.time() - t_start,
                                                  np.round(degs[:6], 1)))
            else:
                try:
                    d, q_new = robot.step(degs)
                    fails = 0
                except Exception as exc:
                    fails += 1
                    print("\n[loop] ⚠️ 通信失败 %d/%d: %s" % (fails, MAX_FAILS, type(exc).__name__))
                    if fails >= MAX_FAILS:
                        LOOP_ALIVE["err"] = "连续通信失败"
                        print("[loop] 连续失败，停止并退出")
                        break
                    time.sleep(0.05)
                    continue
                r, p, _ = imu.update(d["roll"], d["pitch"], d["yaw"], dt)
                if abs(math.degrees(r)) > TILT_ABORT_DEG or abs(math.degrees(p)) > TILT_ABORT_DEG:
                    print("[loop] ⚠️ 倾角超限 roll=%.1f pitch=%.1f，急停"
                          % (math.degrees(r), math.degrees(p)))
                    break
                q_prev = q_new
                if n % 20 == 0 and not KEY_MODE["on"]:
                    print("[run] t=%5.1fs roll=%+5.1f° pitch=%+5.1f° cmd=[%.2f %.2f]"
                          % (time.time() - t_start, math.degrees(r),
                             math.degrees(p), cmd[0], cmd[2]))

            n += 1
            periods.append(time.time() - loop0)
            remain = dt_nom - (time.time() - loop0)
            if remain > 0:
                time.sleep(remain)
    finally:
        LOOP_ALIVE["ok"] = False
        if not dry:
            robot.stop()
        m, p95 = robot.rtt_stats()
        print("\n[loop] %d 帧, 平均 %.1f Hz, RTT 均值 %.1f ms / P95 %.1f ms"
              % (n, n / max(1e-6, time.time() - t_start), m, p95))
        if periods:
            a = np.asarray(periods) * 1000
            print("[loop] 周期 均值 %.1f ms / P95 %.1f ms" % (a.mean(), np.percentile(a, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.11.103")
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cmd", action="store_true", help="开 HTTP 指令接口 8081（给网页用）")
    ap.add_argument("--keys", action="store_true", help="这个终端里直接用键盘遥控")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--policy", default=POLICY)
    args = ap.parse_args()

    robot = RemoteRobot(args.host, args.port)
    try:
        d = robot.state()
        print("[init] 狗端状态 OK: roll=%.1f pitch=%.1f" % (d["roll"], d["pitch"]))
    except Exception as e:
        print("连不上狗端桥接服务 %s:%d → %s" % (args.host, args.port, e))
        print("先在狗上跑: python3 dog_bridge.py --port 8110")
        return 1

    policy = PolicyRunner(args.policy)

    if args.cmd:
        threading.Thread(target=start_cmd_server, args=(8081,), daemon=True).start()
        time.sleep(0.8)

    if args.keys and args.run:
        # 策略循环丢到后台线程，主线程处理键盘
        KEY_MODE["on"] = True
        th = threading.Thread(target=loop, args=(robot, policy, args.seconds), daemon=True)
        th.start()
        time.sleep(1.0)
        keyboard_control()
        robot.stop()
        return 0

    if args.dry:
        loop(robot, policy, args.seconds, dry=True)
    elif args.run:
        loop(robot, policy, args.seconds, dry=False)
    else:
        print("请指定 --dry 或 --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
