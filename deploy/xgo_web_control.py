#!/usr/bin/env python3
"""Loopback-only browser teleoperation for the Mini2S suspended bench test."""

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
import signal
import threading
import time

from flask import Flask, abort, jsonify, render_template_string, request
import numpy as np

from onnx_runner import (
    ACTION_SCALE, CTRL_HZ, DEFAULT_POS, LEG_TARGET_LIMITS, POLICY,
    Imu, JointFeedback, PolicyRunner, Robot, build_obs,
)

DIRECTIONS = {
    "forward": (0.22, 0.0, 0.0),
    "backward": (-0.22, 0.0, 0.0),
    "left": (0.0, 0.0, 0.25),
    "right": (0.0, 0.0, -0.25),
}
# 训练指令范围是前后 −0.30–0.35 m/s、转向 ±0.45 rad/s；
# 上限取 1.25×（后退 −0.275 m/s、转向 0.3125 rad/s），不越出训练范围。
SPEED_SCALES = (0.5, 0.75, 1.0, 1.25)
LEASE_S = 0.32
MAX_HOLD_S = 2.0
TILT_LIMIT_DEG = 12.0
TARGET_RANGE_DEG = 6.0
READ_RANGE_DEG = 10.0
SLEW_DEG = 2.0

# 行走模式：桌面那套 ±6° 包络会把策略 30° 峰峰的步态削成 4–8°，
# 地面上表现为四腿撑住不动。行走模式改回舵机实测限位，
# slew 提到 4°/tick（40 Hz 下 160°/s），单次最长 6 秒。
WALK_SLEW_DEG = 4.0
WALK_HOLD_S = 6.0
WALK_TILT_LIMIT_DEG = 25.0
WALK_READ_RANGE_DEG = 45.0


@dataclass
class Hold:
    token: str
    direction: str
    expires: float
    started: float | None = None
    scale: float = 1.0


class Teleop:
    def __init__(self, policy, robot_factory=Robot, reference_deg=None, walk=False):
        self.policy = policy
        self.walk = walk
        self.slew = math.radians(WALK_SLEW_DEG if walk else SLEW_DEG)
        self.max_hold = WALK_HOLD_S if walk else MAX_HOLD_S
        self.tilt_limit = WALK_TILT_LIMIT_DEG if walk else TILT_LIMIT_DEG
        self.read_range = WALK_READ_RANGE_DEG if walk else READ_RANGE_DEG
        self.robot_factory = robot_factory
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closing = threading.Event()
        self.hold = None
        self.running = False
        self.error = None
        self.sent = 0
        self.tilt = None
        self.track = None
        self.scale = 1.0
        with self.open_robot(read_only=True) as robot:
            self.reference = robot.read_joints().copy()
            self.tilt = self.check_tilt(robot.read_roll(), robot.read_pitch())
        if self.walk:
            # 行走模式不设姿态参考点，直接放开到舵机实测限位
            self.lower = np.array(LEG_TARGET_LIMITS[:, 0], dtype=np.float64)
            self.upper = np.array(LEG_TARGET_LIMITS[:, 1], dtype=np.float64)
        else:
            if reference_deg is not None:
                reference_deg = np.asarray(reference_deg, dtype=np.float64)
                if reference_deg.shape != (12,) or not np.isfinite(reference_deg).all():
                    raise ValueError("参考文件须提供 12 个有效腿部关节角")
                if np.max(np.abs(np.rad2deg(self.reference[:12]) - reference_deg)) > READ_RANGE_DEG:
                    raise RuntimeError("当前姿态与悬空参考点偏差过大")
                self.reference[:12] = np.deg2rad(reference_deg)
            self.span = math.radians(TARGET_RANGE_DEG)
            ref = self.reference[:12]
            self.lower = np.maximum(ref - self.span, LEG_TARGET_LIMITS[:, 0])
            self.upper = np.minimum(ref + self.span, LEG_TARGET_LIMITS[:, 1])
            if not np.all((ref >= self.lower) & (ref <= self.upper)):
                raise RuntimeError("起始关节位置超出已验证限位")
        self.worker = threading.Thread(target=self.work, name="mini2s-uart", daemon=True)
        self.worker.start()

    def open_robot(self, read_only):
        return closing(self.robot_factory(read_only=read_only))

    def check_tilt(self, roll, pitch):
        tilt = max(abs(roll), abs(pitch))
        if not math.isfinite(tilt) or tilt >= self.tilt_limit:
            raise RuntimeError(f"倾角超限（{tilt:.1f}°），已停止")
        return tilt

    def pending(self):
        """Caller holds the lock."""
        h = self.hold
        now = time.monotonic()
        if h and (now >= h.expires or (h.started is not None and now >= h.started + self.max_hold)):
            self.hold = h = None
        return h

    def prepare(self, direction, scale=1.0):
        with self.lock:
            if self.closing.is_set() or self.running or self.pending():
                return None
            self.error = None
            self.scale = scale
            self.hold = Hold(secrets.token_urlsafe(18), direction,
                             time.monotonic() + LEASE_S, scale=scale)
            return self.hold.token

    def pulse(self, token, *, activate=False):
        with self.lock:
            h = self.pending()
            if h is None or h.token != token or (h.started is None and not activate):
                return False
            now = time.monotonic()
            if h.started is None:
                h.started = now
            h.expires = now + LEASE_S
            self.wake.set()
            return True

    def stop(self, token=None):
        with self.lock:
            if token is None or (self.hold and self.hold.token == token):
                self.hold = None
            self.wake.set()

    def status(self):
        with self.lock:
            h = self.pending()
            return {
                "state": "fault" if self.error else ("moving" if self.running and h else "idle"),
                "direction": h.direction if h and self.running else None,
                "speed": h.scale if h and self.running else self.scale,
                "mode": "walk" if self.walk else "bench",
                "frames": self.sent, "tilt_deg": self.tilt,
                "track_deg": self.track, "error": self.error,
            }

    def work(self):
        while not self.closing.is_set():
            self.wake.wait(timeout=0.25)
            self.wake.clear()
            with self.lock:
                h = self.pending()
                if h is None or h.started is None or self.closing.is_set():
                    continue
                self.running = True
            try:
                self.drive(h)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    print(f"[web-gait] {exc}", flush=True)
            finally:
                with self.lock:
                    self.running = False
                    if self.hold is h:
                        self.hold = None

    def drive(self, hold):
        with self.open_robot(read_only=False) as robot:
            imu = Imu(robot)
            feedback = JointFeedback(robot)
            commanded, _ = feedback.read()
            commanded = commanded.copy()
            self.policy.action.fill(0)
            dt = 1.0 / CTRL_HZ
            max_step = self.slew
            if not self.walk:
                initial = commanded[:12]
                if np.max(np.abs(initial - np.clip(initial, self.lower, self.upper))) > max_step:
                    raise RuntimeError("当前关节位置偏离悬空试跑起点，已停止")
            command = np.array(DIRECTIONS[hold.direction], dtype=np.float32) * hold.scale
            worst = 0.0
            first = self.sent
            origin = commanded[:12].copy()
            cmd_span = read_span = 0.0
            while not self.closing.is_set():
                started = time.monotonic()
                with self.lock:
                    if self.pending() is not hold:
                        break
                roll, pitch, _ = imu.update()
                tilt = self.check_tilt(math.degrees(roll), math.degrees(pitch))
                q, velocity = feedback.read()
                worst = max(worst, float(np.max(np.abs(np.rad2deg(q[:12] - commanded[:12])))))
                read_span = max(read_span, float(np.max(np.abs(np.rad2deg(q[:12] - origin)))))
                joint_delta = np.rad2deg(q[:12] - self.reference[:12])
                if np.max(np.abs(joint_delta)) > self.read_range:
                    raise RuntimeError("读回关节超出允许范围，已停止")
                obs = build_obs(imu, q, velocity, command, self.policy,
                                started - hold.started, self.policy.action)
                action = self.policy.infer(obs)
                if not np.isfinite(action).all():
                    raise RuntimeError("推理输出异常，已停止")
                wanted = np.clip((DEFAULT_POS + ACTION_SCALE * action)[:12], self.lower, self.upper)
                commanded[:12] += np.clip(wanted - commanded[:12], -max_step, max_step)
                commanded[:12] = np.clip(commanded[:12], self.lower, self.upper)
                with self.lock:
                    if self.pending() is not hold or self.closing.is_set():
                        break
                    robot.send_legs(commanded)
                    robot.dog.ser.flush()
                    self.sent += 1
                    self.tilt = round(tilt, 2)
                time.sleep(max(0, dt - (time.monotonic() - started)))
            self.track = round(worst, 2)
            cmd_span = float(np.max(np.abs(np.rad2deg(commanded[:12] - origin))))
            mode = "walk" if self.walk else "bench"
            print(f"[web-gait] {mode} {hold.direction} {hold.scale:.2f}×: {self.sent - first} 帧，"
                  f"指令幅度 {cmd_span:.2f}°，读回幅度 {read_span:.2f}°，"
                  f"最大跟踪误差 {worst:.2f}°，UART 重试 {robot.read_retries} 次", flush=True)

    def close(self):
        self.closing.set()
        self.stop()
        self.worker.join(timeout=1)


def app_for(controller, port):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2048
    secret = secrets.token_urlsafe(24)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    html = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")

    @app.before_request
    def local_only():
        if request.host not in hosts:
            abort(403)
        if request.method == "POST" and (
            request.headers.get("Origin") != f"http://{request.host}"
            or not secrets.compare_digest(request.headers.get("X-XGO-Control", ""), secret)
        ):
            abort(403)

    @app.after_request
    def prevent_caching(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/")
    def page():
        return render_template_string(html, control_token=secret, directions=DIRECTIONS,
                                      speeds=SPEED_SCALES, walk=controller.walk,
                                      hold_s=int(controller.max_hold))

    @app.get("/api/status")
    def status():
        return jsonify(controller.status())

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            abort(400)
        return value

    @app.post("/api/prepare")
    def prepare():
        value = body()
        direction = value.get("direction")
        scale = value.get("speed", 1.0)
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            abort(400)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale not in SPEED_SCALES:
            abort(400)
        token = controller.prepare(direction, float(scale))
        return (jsonify(token=token), 200) if token else (jsonify(error="控制忙"), 409)

    @app.post("/api/pulse")
    def pulse():
        value = body()
        ok = controller.pulse(value.get("token"), activate=value.get("activate") is True)
        return jsonify(ok=ok), 200 if ok else 409

    @app.post("/api/stop")
    def stop():
        controller.stop(body().get("token"))
        return jsonify(ok=True)

    return app


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=18107)
    cli.add_argument("--policy", default=POLICY)
    cli.add_argument("--bench-reference", type=Path, help="Reuse a verified bench trial's final pose")
    cli.add_argument("--walk", action="store_true",
                     help="Ground walking: full servo limits, 4 deg/tick slew, 6 s holds")
    args = cli.parse_args()
    if args.walk and args.bench_reference is not None:
        raise SystemExit("--walk 不能用 --bench-reference")
    reference = None
    if args.bench_reference is not None:
        result = json.loads(args.bench_reference.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError("仅可使用已完成的悬空试跑记录")
        reference = result["final_deg"]
    controller = Teleop(PolicyRunner(args.policy), reference_deg=reference, walk=args.walk)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        print(f"[web-gait] Ready on http://127.0.0.1:{args.port}, "
              f"{'walk' if args.walk else 'bench'} mode, no servo output", flush=True)
        app_for(controller, args.port).run(host="127.0.0.1", port=args.port, threaded=True,
                                           debug=False, use_reloader=False)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
