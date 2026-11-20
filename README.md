# luwu_xgo_rl

陆吾 Mini2S 四足机器狗的平地行走强化学习：在 MuJoCo Warp + RSL-RL 上训练速度跟踪策略，导出 ONNX 后在树莓派上用 ONNX Runtime 部署，并带一个只绑定回环地址的网页点动/行走控制页。

训练、验收和实机部署的数据都来自 2026-11-20 这台机器狗，原始记录在 [`results/`](results/)。

## 目录

```
.
├── train/                  # 训练入口与两次训练的完整配置
│   ├── xgo_low_friction.py # 注册 Mini2S-Walk-LowFriction 任务并启动 train/play
│   └── configs/            # 从 4522 取回的 params/env.yaml、agent.yaml
├── eval/                   # 固定摩擦、固定时钟的 checkpoint 验收
├── deploy/                 # 狗端推理与网页控制
│   ├── onnx_runner.py      # 独占 UART、只读推理，--dry / --run / --probe
│   ├── xgo_web_control.py  # 网页点动与地面行走（:18107，仅回环）
│   ├── xgo_web_control.html
│   ├── bench_drive.py      # 走 JSON 接口的短点动复现脚本
│   └── models/             # 部署用 ONNX
├── tools/                  # 实机观测/动作限位画像
├── docs/                   # 训练与部署全流程指南
└── results/                # 验收数据（见 results/README.md）
```

## 环境

训练服务器：Ubuntu 22.04、RTX 4090、Python 3.11、mjlab 1.6.0、MuJoCo Warp 3.11.0、RSL-RL 5.4.2。需要厂商仓库 [LuwuDynamics/luwu_mjlab](https://github.com/LuwuDynamics/luwu_mjlab) 提供 `src.tasks` 和仿真模型。

机器狗端：树莓派 CM5、Debian 13、Python 3.13、`xgolib`、onnxruntime 1.26、固件 `M-7.0.0b8`。

## 训练

`train/xgo_low_friction.py` 要放在 `luwu_mjlab-main` 根目录运行，它依赖该仓库的 `src.tasks` 和 `scripts` 包：

```bash
cd ~/ldy/luwu_mjlab-main
cp /path/to/luwu_xgo_rl/train/xgo_low_friction.py .
PY="$HOME/ldy/miniconda3/envs/luwu/bin/python"

"$PY" -u xgo_low_friction.py train --env.scene.num-envs 32  --agent.max-iterations 2   --agent.run-name smoke
"$PY" -u xgo_low_friction.py train --env.scene.num-envs 1024 --agent.max-iterations 900 --agent.run-name low_traction_1024
```

每个并行环境在 μ=0.07–0.80 对数均匀采样，同时写入地面和四条小腿碰撞体；指令范围前后 −0.30–0.35 m/s、横向 ±0.16 m/s、转向 ±0.45 rad/s。两次训练的完整配置见 `train/configs/`。

## 验收

`eval/evaluate_xgo_traction.py` 在同一组初始姿态和指令下跑固定摩擦矩阵（默认 μ=0.08、0.12、0.20、0.35、0.70），支持 `--timing-hz 40/50` 模拟策略频率、`--hardware-feedback-steps 1/2/3` 模拟实机 25/50/75 ms 的姿态更新间隔：

```bash
"$PY" -u evaluate_xgo_traction.py \
  --checkpoint logs/rsl_rl/.../model_449.pt \
  --onnx logs/rsl_rl/.../policy.onnx \
  --out results/traction_eval.json
```

## 部署

狗端用 `deploy/onnx_runner.py`，它绕开会下发初始化动作的 SDK 构造函数，直接以独占方式打开 `/dev/ttyAMA0`：

```bash
python3 -u onnx_runner.py --dry --seconds 5      # 只读推理，不下发
python3 -u onnx_runner.py --probe 31 3           # 单舵机点动 ±5° 内
python3 -u onnx_runner.py --run                  # 闭环行走，指令走 /cmd
```

网页控制（默认桌面/悬空模式）：

```bash
python3 -u xgo_web_control.py --port 18107
ssh -N -L 18107:127.0.0.1:18107 xgo-mini2s        # 在 Mac 上开隧道
```

`--walk` 切到地面行走模式。两者的差别：

| | 桌面模式（默认） | 行走模式 `--walk` |
|---|---|---|
| 目标包络 | 参考姿态 ±6° | 舵机实测限位 |
| 每周期最大变化 | 2° | 4° |
| 单次最长 | 2 s | 6 s |
| 倾角保护 | 12° | 25° |

页面只有前、后、左、右四个方向，按住才走，心跳 80 ms 续租、松手 0.32 s 内停。指令速度分 50%/75%/100%/125% 四档（数字键 1–4），上限取 125% 是为了后退 −0.275 m/s、转向 0.3125 rad/s 不越出训练指令范围。

## 这次调试得到的结论

**“往前走不动”的根因是限幅，不是策略。** 同一份 ONNX 开环回放，前向 0.22 m/s 时 thigh 目标峰峰 12–25°、calf 22–32°；桌面模式的 ±6° 包络和 2°/tick 斜率把实际目标削到 thigh 4°、calf 4–8°，地面上就是四腿撑住不动。切成 `--walk` 后 1.0× 前进 1.5 秒（51 帧）倾角只有 2–6°，现场确认能往前走。

**起步有个坑。** 狗蹲着（thigh 约 35°）时直接按前进，策略会先把四条腿撑到 57° 左右的默认姿态，同时起振容易前倾，实测倾斜到 28.1° 触发保护。先让它空站一个周期再按方向键。

**UART 读加了重发。** 之前每次试跑约 50 帧会以 `Invalid UART feedback for ROLL` 终止，`_read_register` 改成每次最多重试 3 次（单次超时 30 ms）后，29 次桌面试跑 1552 帧没有再失败，重试计数也是 0——原故障没有复现，重发只是兜底。空闲状态另外做过 5000 次只读压力测试，同样 0 失败。

**关节映射。** 策略顺序 fl/fr/br/bl 每条腿 hip、thigh、calf，对应舵机 `13/12/11、23/22/21、33/32/31、43/42/41`，SDK 的角度反馈在每条腿内是反序的。限幅取 hip ±31°、thigh −66°–90°、calf −73°–57°。

## 已知问题

- 关节位置跟踪偏弱：桌面记录里读回幅度只有指令的 1/3–2/3，单帧最大误差 8–10°。行走模式没有改观测通道，这部分仍待查。
- `dog.read_imu()` 返回死数据，`base_ang_vel` 只能用姿态角差分估计；`yaw` 静止时固件不更新。
- 实机闭环依赖 `/dev/ttyAMA0` 独占，跑之前要确认没有别的进程占着串口。

完整流程和逐项数据见 [训练指南](docs/Luwu%20XGO%20强化学习训练指南.md)。
