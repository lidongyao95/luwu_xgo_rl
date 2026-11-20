#!/usr/bin/env python3
"""对 xgo_web_control 的 HTTP 接口做一次短点动，用于桌面试跑取证。

在 Mac 上先开隧道：ssh -N -L 18107:127.0.0.1:18107 xgo-mini2s
用法：python3 bench_drive.py forward 1.9 [速度倍数，如 0.5 / 1.0 / 1.25]
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:18107"


def req(path, payload=None, secret=None):
    data = None if payload is None else json.dumps(payload).encode()
    r = urllib.request.Request(BASE + path, data=data,
                               method="POST" if payload is not None else "GET")
    r.add_header("Origin", BASE)
    if payload is not None:
        r.add_header("Content-Type", "application/json")
        r.add_header("X-XGO-Control", secret)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def main():
    direction = sys.argv[1] if len(sys.argv) > 1 else "forward"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 1.8
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    page = urllib.request.urlopen(BASE + "/", timeout=5).read().decode()
    secret = json.loads(re.search(r"const secret = (\"[^\"]*\")", page).group(1))

    code, body = req("/api/prepare", {"direction": direction, "speed": speed}, secret)
    print("prepare", code, body, flush=True)
    if code != 200:
        return 1
    token = body["token"]
    t0 = time.monotonic()
    last = 0.0
    frames0 = req("/api/status")[1]["frames"]
    while time.monotonic() - t0 < seconds:
        code, body = req("/api/pulse", {"token": token, "activate": True}, secret)
        if code != 200:
            print("pulse rejected", code, body, flush=True)
            break
        now = time.monotonic()
        if now - last > 0.25:
            last = now
            print("t=%.2fs status=%s" % (now - t0, req("/api/status")[1]), flush=True)
        time.sleep(0.08)
    req("/api/stop", {"token": token}, secret)
    time.sleep(0.4)
    final = req("/api/status")[1]
    print("final", final, "frames_delta", final["frames"] - frames0, flush=True)
    return 0 if final["error"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
