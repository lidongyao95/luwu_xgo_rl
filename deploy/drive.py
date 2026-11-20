#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGO 策略终端遥控器 —— 在 Mac 终端里用键盘实时驱动真机上的 ONNX 策略

用法：
    python3 drive.py                     # 连 172.18.11.103:8100
    python3 drive.py --script forward    # 脚本化测试：前进 2 秒（不用键盘）

键盘：
    W / ↑    前进        S / ↓    后退
    A / ←    左转        D / →    右转
    空格      停车        Q        退出

注意：狗的 ONNX 推理进程必须在跑（狗上 onnx_runner.py --run --serve-cmd 8100）。
      推理在狗本地跑，控制循环 40 Hz；本机只负责把稀疏的按键指令发过去。
"""

import argparse
import select
import sys
import termios
import time
import tty

import requests

VX, VYAW = 0.35, 0.8          # 策略单位：m/s, rad/s
SEND_HZ = 15.0


class Driver:
    fail_streak = 0

    def __init__(self, host, port, vx=VX, vyaw=VYAW):
        self.base = "http://%s:%d" % (host, port)
        self.vx, self.vyaw = vx, vyaw
        self.last = {"vx": 0.0, "vyaw": 0.0}

    def send(self, vx, vyaw):
        if (vx, vyaw) == (self.last["vx"], self.last["vyaw"]):
            return
        try:
            requests.get(self.base + "/cmd", params={"vx": vx, "vyaw": vyaw}, timeout=2.0)
            self.last = {"vx": vx, "vyaw": vyaw}
            Driver.fail_streak = 0
        except Exception:
            Driver.fail_streak += 1
            # 只在连续失败时提示一次，避免刷屏
            if Driver.fail_streak in (1, 5, 20):
                print("\n[!] 连接狗失败（连续 %d 次），检查狗是否掉线" % Driver.fail_streak)

    def stop(self):
        try:
            requests.get(self.base + "/stop", timeout=2.0)
        except Exception:
            pass
        self.last = {"vx": 0.0, "vyaw": 0.0}

    def keys_to_cmd(self, held):
        vx = vyaw = 0.0
        if "w" in held:
            vx += self.vx
        if "s" in held:
            vx -= self.vx
        if "a" in held:
            vyaw += self.vyaw
        if "d" in held:
            vyaw -= self.vyaw
        return vx, vyaw


def read_keys(timeout):
    """非阻塞读一个字符，没有就返回 None。"""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    return sys.stdin.read(1)


KEY_TIMEOUT = 0.30      # 这么久没收到该键的重复事件，就当松开了


def run_keyboard(drv):
    """终端键盘遥控。

    关键点：终端只有「按下」事件，没有「松开」事件。所以用键盘自动重复
    （按住时会连续发同一个字符）来判断：超过 KEY_TIMEOUT 没再收到，就认为松开了。
    早期版本没有这个超时，按一下 W 就会一直走下去。
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    held = {}                      # key -> 最后一次收到的时间
    print(__doc__)
    print("已连接 %s\n按 W/A/S/D 驱动，空格停，Q 退出\n" % drv.base)
    try:
        tty.setcbreak(fd)
        last_tx = 0.0
        while True:
            ch = read_keys(0.01)
            now = time.time()

            if ch:
                if ch in ("q", "Q"):
                    break
                if ch == " ":
                    held.clear()
                    drv.stop()
                    print("\r停车                              ", end="", flush=True)
                    continue
                if ch == "\x1b":                 # 方向键 ESC [ X
                    read_keys(0.01)
                    third = read_keys(0.01)
                    ch = {"A": "w", "B": "s", "C": "d", "D": "a"}.get(third, "")
                ch = ch.lower()
                if ch in "wasd":
                    held[ch] = now

            # 超过 KEY_TIMEOUT 没重复，判为松开
            for k in [k for k, t in held.items() if now - t > KEY_TIMEOUT]:
                del held[k]

            vx, vyaw = drv.keys_to_cmd(held)
            if now - last_tx > 1.0 / SEND_HZ:
                drv.send(vx, vyaw)
                last_tx = now
                print("\r  %s  vx=%+.2f  wyaw=%+.2f   " %
                      ("".join(sorted(held)) or "·", vx, vyaw), end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        drv.stop()
        print("\n已停车，终端已恢复")


def run_repl(drv):
    """不用原始模式，逐行输入指令 —— 任何终端都能用。

        w 0.5     前进 0.5 秒
        s 0.5     后退
        a 0.5     左转
        d 0.5     右转
        x         停车
        q         退出
    """
    print("指令格式: w/s/a/d + 秒数，例如 'w 0.5'；x 停，q 退出")
    while True:
        try:
            line = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break
        if line in ("x", "stop", "s0"):
            drv.stop()
            print("  停车")
            continue
        parts = line.split()
        key = parts[0]
        dur = float(parts[1]) if len(parts) > 1 else 0.4
        if key not in ("w", "s", "a", "d"):
            print("  无效指令")
            continue
        vx, vyaw = drv.keys_to_cmd({key})
        drv.send(vx, vyaw)
        time.sleep(dur)
        drv.stop()
        print("  %s %.2fs 完成" % (key, dur))
    drv.stop()
    print("已退出")


def run_script(drv, name, seconds=2.0):
    seq = {
        "forward": (drv.vx, 0.0),
        "back": (-drv.vx, 0.0),
        "left": (0.0, drv.vyaw),
        "right": (0.0, -drv.vyaw),
        "still": (0.0, 0.0),
    }[name]
    print("[script] %s  %s  持续 %.1fs" % (name, seq, seconds))
    drv.send(*seq)
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            time.sleep(0.1)
    finally:
        drv.stop()
        print("[script] 已停")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.11.103")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--script", choices=["forward", "back", "left", "right", "still"])
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--repl", action="store_true",
                    help="逐行输入指令，不依赖终端的原始模式")
    args = ap.parse_args()

    drv = Driver(args.host, args.port)
    # 连通性自检
    try:
        requests.get(drv.base + "/stop", timeout=2.0)
    except Exception as e:
        print("连不上策略进程 %s: %s" % (drv.base, e))
        print("先在狗上跑: python3 onnx_runner.py --run --serve-cmd 8100")
        return 1

    if args.script:
        run_script(drv, args.script, args.seconds)
    elif args.repl:
        run_repl(drv)
    else:
        run_keyboard(drv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
