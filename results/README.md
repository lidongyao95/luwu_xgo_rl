# 验收数据（2026-11-20）

全部是原始 JSON，字段名即含义。仿真数据在 4522 用 `eval/evaluate_xgo_traction.py` 生成，实机数据在树莓派上用 `deploy/` 下的脚本采集。

## 仿真

| 文件 | 内容 |
|---|---|
| [`traction_eval.json`](2026-11-20/traction_eval.json) | 固定摩擦矩阵（μ=0.08–0.70）下三个 checkpoint 的站立/前进/转向验收，含 ONNX 与 PyTorch 的一致性比对 |
| [`traction_timing_40hz.json`](2026-11-20/traction_timing_40hz.json) | 同一矩阵在 40 Hz 策略频率下的复测 |
| [`traction_timing_50hz.json`](2026-11-20/traction_timing_50hz.json) | 同一矩阵在 50 Hz 策略频率下的复测 |
| [`hardware-feedback/replay-correct-servo-limits.json`](2026-11-20/hardware-feedback/replay-correct-servo-limits.json) | 换上修正后的舵机限幅、25 ms 姿态更新间隔的回放（160/160 完成） |
| [`hardware-feedback/hardware_25ms.json`](2026-11-20/hardware-feedback/hardware_25ms.json) | 实机反馈适配版在 25 ms 姿态更新间隔下的验收 |
| [`hardware-feedback/hardware_75ms.json`](2026-11-20/hardware-feedback/hardware_75ms.json) | 同上，75 ms 间隔 |

## 实机

| 文件 | 内容 |
|---|---|
| [`hardware-feedback/uart_readonly.json`](2026-11-20/hardware-feedback/uart_readonly.json) | 只读串口采样：固件、电量、姿态与关节寄存器读取耗时分布 |
| [`hardware-feedback/onnx_readonly_on_robot.json`](2026-11-20/hardware-feedback/onnx_readonly_on_robot.json) | 狗端 ONNX Runtime 只读推理 80 帧，未下发任何舵机指令 |
| [`hardware-feedback/forward-backward-readonly.json`](2026-11-20/hardware-feedback/forward-backward-readonly.json) | 前进/后退指令下的只读姿态与关节记录 |
| [`hardware-feedback/probe-31-aba.json`](2026-11-20/hardware-feedback/probe-31-aba.json) | 31 号舵机单帧 A→B→A 点动，指令与读回对照 |
| [`hardware-feedback/probe-31-aba-refreshed.json`](2026-11-20/hardware-feedback/probe-31-aba-refreshed.json) | 同上，改为持续刷新指令 |
| [`hardware-feedback/probe-33-control.json`](2026-11-20/hardware-feedback/probe-33-control.json) | 33 号舵机对照组 |
| [`hardware-feedback/first-suspended-policy-run.json`](2026-11-20/hardware-feedback/first-suspended-policy-run.json) | 首次悬空闭环试跑：48 帧的指令/读回/倾角，`final_deg` 被网页控制的 `--bench-reference` 复用 |

训练日志、TensorBoard events 和 `model_*.pt` checkpoint 没有入库（单个 run 30–50 MB），只在 4522 的 `~/ldy/luwu_mjlab-main/logs/rsl_rl/` 下保留。
