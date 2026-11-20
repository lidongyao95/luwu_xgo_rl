# Luwu XGO 强化学习训练指南

适用任务为 XGO Mini2S 平地步态，机械臂在仿真模型中占 3 个自由度。训练用 MuJoCo Warp 和 RSL-RL PPO，目标是在低摩擦平面上跟踪身体坐标系的前进、横移和转向速度。服务器上的任务名为 `Mini2S-Walk-LowFriction`。

实机型号经现场确认为 MINI2S，固件为 `M-7.0.0b8`，可通过 `xgolib` 进行串口读数。本文采用 XGO Mini2S 仿真模型和部署映射。德克仕与这台实机的供货关系待采购信息核对。舵机编号和方向逐项以实机动件和读回结果验收。

## 1. 连接 4522 并检查环境

SSH 连接信息来自 `remote-test/docs/jushenzhineng-xrdp-client-runbook.md`。从 Mac 连接后，所有训练和仿真命令在服务器的 `~/ldy/luwu_mjlab-main` 执行。

```bash
ssh -p 4522 sjtu@202.121.181.124
cd ~/ldy/luwu_mjlab-main
PY="$HOME/ldy/miniconda3/envs/luwu/bin/python"
nvidia-smi
"$PY" -V
"$PY" -c 'import torch, mujoco; print(torch.cuda.is_available(), mujoco.__version__)'
"$PY" scripts/list_envs.py --keyword Mini2S
```

本次使用 Ubuntu 22.04.5、RTX 4090、Python 3.11.18、mjlab 1.6.0、MuJoCo / MuJoCo Warp 3.11.0、Warp 1.17.0、PyTorch 2.15.0 和 RSL-RL 5.4.2。`list_envs.py` 列出厂商任务，低摩擦任务由本文的入口脚本注册。首次在其他服务器配置时，可按厂商仓库的 Linux 部分安装，并用上面的命令确认模拟栈及 CUDA。现有 4522 环境可直接复用。 [Luwu 官方训练代码与安装说明](https://github.com/LuwuDynamics/luwu_mjlab)

如果换了服务器或重新解包，在 Mac 上从 `RL` 目录上传这两个文件。4522 上已放好。

```bash
scp -P 4522 train/xgo_low_friction.py eval/evaluate_xgo_traction.py \
  sjtu@202.121.181.124:ldy/luwu_mjlab-main/
```

## 2. 低摩擦是怎么进入仿真的

没有显式接触对时，MuJoCo 使用优先级较高一侧的摩擦系数。两侧优先级相同时取较大值。原始 Mini2S 的四只脚优先级是 1，地面优先级是 0。 [MuJoCo 接触参数](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)

按 11 月 18 日 15:29 的碰撞优先级在 4522 复测，只把脚设为 μ=0.08、地面保留 μ=1.00，24 个求解器足地接触的两条切向系数都为 0.08。脚和地面都设为 0.08 时，23 个接触也全部为 0.08。该次原始训练的脚部随机化范围为 μ=0.20–3.00，因此训练采样范围本身不包含 μ=0.08。

17:16 的 `~/ldy/train_lowfric.py` 用补丁将足地摩擦设为同一个 μ=0.06–0.35 的均匀随机值。诊断时把旧补丁固定在 μ=0.08，读取到的两条切向系数均为 0.08。补丁也误选了机械臂的 `arm_calf_collision`，异常发生后会继续训练，保存的 `env.yaml` 只记录了改写前的脚部和地面范围。两份旧模型可以复用为对照。

新任务使用 mjlab 启动事件，为每个并行环境从 μ=0.07–0.80 对数均匀采样一个值，将其同时写入地面和四条腿的小腿碰撞体。每个世界的系数在训练期间固定，机械臂碰撞体保持 MJCF 原值，范围随 `env.yaml` 保存。固定 μ=0.08 的回归检查同时验证了四只脚与地面的实际接触系数和机械臂原值。

训练速度范围为前后 −0.30–0.35 m/s、横向 ±0.16 m/s、转向 ±0.45 rad/s。配置另有 10% 站立环境，足端滑移惩罚权重为 −0.25，随机扰动幅度按低速场景收窄。这些是本次试训的设置，速度范围与扰动幅度需要结合使用场地验证。

## 3. 冒烟测试和正式训练

```bash
cd ~/ldy/luwu_mjlab-main
PY="$HOME/ldy/miniconda3/envs/luwu/bin/python"

"$PY" -u xgo_low_friction.py train \
  --env.scene.num-envs 32 \
  --agent.max-iterations 2 \
  --agent.run-name smoke_traction

"$PY" -u xgo_low_friction.py train \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 900 \
  --agent.run-name low_traction_1024
```

检查冒烟测试是否生成 `model_1.pt`、`policy.onnx` 和 `params/env.yaml`。正式训练可用 `nohup` 运行，给每次启动使用独立的日志文件，并在命令退出后记录退出码。本次日志入口为 `~/ldy/xgo-low-traction-20261120/train.log`。

```bash
tail -n 30 ~/ldy/xgo-low-traction-20261120/train.log
cat ~/ldy/xgo-low-traction-20261120/train.exit
ls -dt logs/rsl_rl/mini2s_velocity_walk_low_traction/*/
```

`policy.onnx` 会随模型保存而覆盖。训练结束且退出码为 0 后，将最后一个 `model_*.pt` 与同目录的 `policy.onnx` 一起验收。TensorBoard 可读 `logs/rsl_rl/mini2s_velocity_walk_low_traction`，训练奖励只用于看收敛趋势，光滑平面性能用下一节的固定摩擦回放判断。

## 4. 固定摩擦验收

评估器在同一组初始姿态和指令下运行所有 checkpoint。默认取 μ=0.08、0.12、0.20、0.35、0.70，各跑站立、前进 0.15 m/s、前进 0.25 m/s、原地转向 0.35 rad/s。每格 8 次，每次 8 秒，共 160 次/模型。评估时关掉观测噪声、训练时的关节反馈和执行延迟，以及随机推扰，保留碰撞与倒地终止。脚本先核对每个并行世界的求解器足地接触，之后记录完整跑完次数、速度误差、足端接触滑移和位移。速度与滑移统计跳过开始的 1 秒，发生终止的试验从首次终止时停止累计。

```bash
cd ~/ldy/luwu_mjlab-main
PY="$HOME/ldy/miniconda3/envs/luwu/bin/python"
OLD=logs/rsl_rl/mini2s_velocity_walk
RUN=$(ls -dt logs/rsl_rl/mini2s_velocity_walk_low_traction/*_low_traction_1024 | head -1)
ORT="$HOME/ldy/xgo-low-traction-20261120/ort"
test -f "$RUN/model_899.pt" && test -f "$RUN/policy.onnx" &&
  PYTHONPATH="$ORT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -u evaluate_xgo_traction.py \
    --checkpoint "$OLD/2026-11-18_15-29-02/model_1499.pt" \
    --checkpoint "$OLD/2026-11-18_17-16-36/model_1499.pt" \
    --checkpoint "$RUN/model_899.pt" \
    --onnx "$RUN/policy.onnx" \
    --out "$RUN/traction_eval.json"
```

`--onnx` 比较最后一份 checkpoint 和 ONNX 在实际回放观测上的动作。4522 的 `ORT` 目录已安装 ONNX Runtime 1.32.0，回放使用 CPUExecutionProvider。没有 ONNX Runtime 的环境会使用 ONNX 自带的参考执行器。沿用上面的命令并加上 `--timing-hz 40` 或 `--timing-hz 50`，可以把物理积分步长都设为 5 ms，分别以 25 ms 或 20 ms 的策略周期评估。用 `--friction 0.03 0.05 0.08 0.12 0.35` 增加低于训练范围的测试，每次给 `--out` 指定独立的结果文件。

## 5. 策略接口与实机差异

本次训练控制周期为 `0.004 s × 5 = 0.020 s`（50 Hz），步态相位周期为 0.4 s。ONNX 输入 `obs[1,56]` 依次是机体角速度、投影重力、速度指令、相位、相对关节位置、关节速度和上一步动作。输入单位用 rad、rad/s 和 m/s，输入归一化在模型内。输出 `actions[1,15]`，目标角度为默认关节角加 `0.25 × action`。按 ONNX 元数据读取关节顺序和默认角度。 [Luwu 官方训练与部署教程](https://wiki.xgorobot.com/kb/the-xgo-series-robot/cmqcbagh700138ffsq1nqs63g?model=mini2s)

Nav 目录中原有 ONNX 的 SHA-256 与 11 月 18 日 15:29 的模型一致。现有狗端脚本 `Nav/src/robot/gait/onnx_runner.py` 使用 40 Hz 名义周期，角速度以姿态差分和低通估计，只驱动 12 个腿部关节。向旧脚本的姿态处理逻辑输入 +179° → −179°，25 ms 后估计的 yaw 角速度为 −29.99 rad/s。将角差折返为 +2° 后，相同滤波条件得到 +0.168 rad/s。当天的硬件体检也记录了原始 IMU 包失效和 yaw 死区。部署验收还需核对实测控制周期、舵机关节映射与方向、角速度和机械臂姿态。本次执行范围为 4522 上的仿真和 ONNX 推理。 [机器狗硬件体检](../Nav/docs/reports/机器狗硬件体检-2026-11-20.md)

## 6. 2026-11-20 验证记录

4522 上的正式训练完成 900 轮，使用 1024 个并行环境、每轮每环境 24 步、训练种子 42，训练循环耗时 15 分 17 秒，退出码为 0。64 环境抽样检查测得 μ=0.0760–0.7364，432 个求解器足地接触的系数与各环境采样值一致，重置后仍保持，机械臂碰撞体的切向系数为 1.0。

服务器产物位于 `~/ldy/luwu_mjlab-main/logs/rsl_rl/mini2s_velocity_walk_low_traction/2026-11-20_14-53-52_low_traction_1024/`，其中最终权重为 `model_899.pt`。[本地 ONNX](../deploy/models/mini2s_walk_low_traction.onnx)通过模型结构检查，并用 ONNX Runtime CPU 在 60 帧回放观测上与 PyTorch 对照，最大单动作绝对误差为 5.96 × 10⁻⁷。ONNX 的 SHA-256 为 `f66f59e240e05223afb625820bfcba8842af55489e738bf8e5d3bb01363ee7b0`。

默认回放核对了 1,356 个求解器足地接触。表中前进误差和足端滑移是 μ=0.08、0.12、0.20 下两档前进指令的均值。转向漂移是在 μ=0.08 下执行 0.35 rad/s 原地转向后，机身相对起点的平均平面位移。

| 模型 | 完成次数 | 前进速度平均绝对误差（m/s） | 前进时足端接触滑移（m/s） | 原地转向漂移（m） |
| --- | ---: | ---: | ---: | ---: |
| 11-18 15:29，Nav 原模型 | 160/160 | 0.061 | 0.065 | 0.040 |
| 11-18 17:16，旧低摩擦模型 | 160/160 | 0.035 | 0.086 | 0.097 |
| 11-20，新低摩擦模型 | 160/160 | 0.040 | 0.081 | 0.008 |

新模型在 μ=0.08 下收到 0.25 m/s 前进指令时，8 次平均实际速度为 0.300 m/s，8 秒平均前进 2.28 m。原地转向漂移变小，足端滑移仍高于 Nav 的原模型。两份旧模型也完成了本次固定摩擦试验，用户之前遇到的实机光滑地面故障有待通过实机日志定位。 [标准回放原始数据](../results/2026-11-20/traction_eval.json)

固定 5 ms 物理步长后，三个模型在 40 Hz 和 50 Hz 下各完成 160/160 次，包括 μ=0.03 和 0.05。最滑的 μ=0.03 下，新模型的实际转速约 0.24 rad/s，指令为 0.35 rad/s。两档策略频率的仿真均未复现旧模型失稳。 [40 Hz 回放数据](../results/2026-11-20/traction_timing_40hz.json) · [50 Hz 回放数据](../results/2026-11-20/traction_timing_50hz.json)

## 7. 针对本机串口反馈的再训练

### 实测与仿真假设

在树莓派上只发送串口读寄存器请求，80 帧均读到了 15 个舵机位置。固件为 `M-7.0.0b8`。115200 baud 下，三个姿态角和关节位置的合计读取时延中位数为 16.93 ms，P95 为 19.21 ms。本次采样时 yaw 静止值保持，三次读取的原始 IMU 数据完全重复且未通过数值合理性检查。先前手摆日志的记录间隔中位数为 62 ms。串口读取耗时和手摆日志间隔各自描述上述测量条件。 [只读串口采样结果](../results/2026-11-20/hardware-feedback/uart_readonly.json)

`Mini2S-Walk-HardwareFeedback` 保留 56 维输入和 15 维输出，以 5 ms 物理步长和 25 ms 控制周期执行。仿真 actor 先采样和保持姿态角，再按实际采样间隔差分，做角度折返、机体系换算和低通滤波。同一次姿态采样用于计算投影重力。训练为每个 episode 选取 25、50 或 75 ms 的姿态更新间隔，并模拟 yaw 保持。yaw 阈值 0.05–0.40° 是训练用的随机化范围，实测数据仅确认了存在保持，未标定阈值。

腿关节沿用仿真反馈延迟，三个机械臂观测量按相对默认值和速度为零处理，仿真机械臂保持默认目标。腿部目标角按当前 MINI 舵机实际编码范围和狗端 4°/周期上限限制。横移指令固定为 0，与当前狗端命令接口一致。真机的实际折臂姿势、舵机零位和映射方向有待单独核对。摩擦沿用第 2 节的设置。

工作区狗端脚本 `Nav/src/robot/gait/onnx_runner.py` 已改为按单调时钟的实测间隔差分，并对 yaw 跨 ±180° 做折返。滤波时间常数与仿真模拟一致。注入假串口数据验证了跨界角速度、间隔变化和腿部关节速度。实机只读验证绕过默认 SDK 构造函数，因为该构造函数本身会下发初始化指令。

### 复现训练与回放

4522 上已部署本任务和评估脚本。正式训练使用第 6 节的低摩擦权重作为初始 actor 和 critic，重新初始化 PPO 优化器。仿真冒烟和正式训练各用独立日志目录。

```bash
cd ~/ldy/luwu_mjlab-main
PY="$HOME/ldy/miniconda3/envs/luwu/bin/python"
BASE="$PWD/logs/rsl_rl/mini2s_velocity_walk_low_traction/2026-11-20_14-53-52_low_traction_1024/model_899.pt"
INIT=logs/rsl_rl/mini2s_walk_hardware_feedback/initial_traction
mkdir -p "$INIT"
test -e "$INIT/model_899.pt" || ln -s "$BASE" "$INIT/model_899.pt"

"$PY" check_xgo_hardware_profile.py
"$PY" -u xgo_hardware_profile.py train \
  --env.scene.num-envs 1024 --agent.max-iterations 450 \
  --agent.run-name hardware_1024 --agent.resume True \
  --agent.load-run '^initial_traction$' --agent.load-checkpoint '^model_899[.]pt$'
```

新旧模型共用同一组起始姿态、摩擦和指令。回放加入狗端对应的 25° roll/pitch 停止阈值，并分别将姿态样本保持 25 ms 或 75 ms，每份模型每个条件组合跑 8 秒。正式训练日志位于 `~/ldy/xgo-hardware-feedback-20261120/train.log`，权重位于 `logs/rsl_rl/mini2s_walk_hardware_feedback`。

```bash
RUN=logs/rsl_rl/mini2s_walk_hardware_feedback/2026-11-20_15-37-29_hardware_1024
for steps in 1 3; do
  "$PY" -u evaluate_xgo_traction.py \
    --checkpoint "$BASE" --checkpoint "$RUN/model_449.pt" \
    --hardware-feedback-steps "$steps" --friction 0.03 0.08 0.12 0.35 0.70 \
    --out "$RUN/hardware_replay_$((steps * 25))ms.json"
done
```

### 2026-11-20 再训练结果

1024 个环境完成 450 轮 PPO，训练循环耗时 8 分 29 秒，退出码为 0。最终权重为 `model_449.pt`，同目录的 `policy.onnx` 经 60 帧实际仿真观测比对，PyTorch 与 CPU ONNX Runtime 的最大单动作绝对误差为 4.77 × 10⁻⁷。本地 ONNX 的 SHA-256 为 `8fed019ec89fc71d672de4b7e0ac52603d6c8338f2c6d849b6ef5aae74d3abaa`。 [实机反馈适配版 ONNX](../deploy/models/mini2s_walk_policy.onnx)

两份模型在 25 ms 和 75 ms 两档姿态更新间隔下，各完成 160/160 次。直行平均绝对误差取五档摩擦和两档前进指令的均值。

| 仿真姿态更新间隔 | 上一版直行误差（m/s） | 新版直行误差（m/s） | 上一版转向误差（rad/s） | 新版转向误差（rad/s） |
| --- | ---: | ---: | ---: | ---: |
| 25 ms | 0.0324 | 0.0280 | 0.0622 | 0.0621 |
| 75 ms | 0.0324 | 0.0288 | 0.0624 | 0.0625 |

最滑的 μ=0.03 在 25 ms 测试下，给定转向 0.35 rad/s，实测从 0.237 增至 0.246 rad/s，机身横平面平均漂移从 0.8 cm 增至 10.6 cm。五档摩擦的直行足端滑移整体接近。 [25 ms 对照数据](../results/2026-11-20/hardware-feedback/hardware_25ms.json) · [75 ms 对照数据](../results/2026-11-20/hardware-feedback/hardware_75ms.json)

树莓派用 ONNX Runtime 1.26.0 读取真实 UART 数据，80 帧全部得到有限动作，舵机指令发送次数为 0。读取、观测计算及推理合计中位耗时为 17.22 ms，纯推理中位耗时为 0.19 ms。本次实机验证覆盖静止姿态的读数与推理，实际行走仍待验收。 [狗端只读 ONNX 结果](../results/2026-11-20/hardware-feedback/onnx_readonly_on_robot.json)

## 8. 狗端部署与悬空点动

新版模型和推理脚本已复制到狗端 `/home/pi/tools/xgo-gait/2026-11-20-hardware-feedback/`，文件校验通过。这里的 `--dry` 通过独占 UART 直接读取寄存器，跳过会下发初始化动作的 SDK 构造流程。固件 `M-7.0.0b8` 返回 16 字节的关节位置寄存器数据，腿部取前 12 字节，再转换成模型的关节顺序。已在已部署的目录下运行 5 秒只读推理，共 200 帧，39.9 Hz，ONNX 平均推理时间 0.11 ms。

```bash
ssh xgo-mini2s
cd ~/tools/xgo-gait/2026-11-20-hardware-feedback
sha256sum -c SHA256SUMS
python3 -u onnx_runner.py --dry --seconds 5
```

模型顺序为每条腿 `hip、thigh、calf`，对应舵机编号是 `13/12/11、23/22/21、33/32/31、43/42/41`。现场悬空点动确认 31 号对应右后腿靠脚的关节。旧脚本按 SDK 原始顺序读写，导致每条腿的第一、第三关节对调。现已修正顺序和各关节限幅。 [陆吾 Mini2S 策略部署映射](https://wiki.xgorobot.com/kb/the-xgo-series-robot/cmqcbagh700138ffsq1nqs63g?model=mini2s)

训练产物沿用第 7 节记录的原始训练设置。部署前将修正后的 `hip ±31°、thigh −66° 至 90°、calf −73° 至 57°` 应用到仿真回放，25 ms 姿态更新条件下，新模型仍完成 160/160 次。 [按正确舵机限幅回放](../results/2026-11-20/hardware-feedback/replay-correct-servo-limits.json)

最后一次悬空点动中，31 号从 −9.78° 指令到 −6.73°，保持阶段的 10 次读回均为 −9.78°。发回原位置指令后，最终读到 −11.82°。现场看到了靠脚关节动作，指令方向与位置跟踪仍有待确认。实机验收目前到已部署版本的只读推理和单舵机位置核对。

### 31 号与 33 号悬空复测

采用 A → B → A，小幅改变一个舵机的目标角，读取每段末尾 5 帧的中位数。先用单帧指令，再用持续刷新指令复测 31 号。33 号用于同一条腿上另一关节的对照。

| 舵机 | 指令方式 | A → B 指令变化 | A → B 读回变化 | 回到 A 后相对第一次 A |
| --- | --- | ---: | ---: | ---: |
| 31，右后腿靠脚 | 单帧 | +2.55° | +0.51° | +0.51° |
| 31，右后腿靠脚 | 持续刷新 | +2.55° | +0.51° | +0.51° |
| 33，右后腿靠机身 | 持续刷新 | +2.43° | +0.49° | −0.24° |

两处的读回变化约为指令变化的五分之一。此结果只覆盖本次姿态下的窄角度范围。持续刷新 31 号并未改善响应，固定位置偏置也不足以解释测得的变化量。当前串口链路尚未通过关节位置跟踪验收，实机闭环尚未开启。陆吾另有基于 `xgo_lib2` 的位置/速度反馈路径，可用于交叉核对。部署机和本次核对的公开仓库版本未找到对应模块。 [31 号单帧记录](../results/2026-11-20/hardware-feedback/probe-31-aba.json) · [31 号刷新记录](../results/2026-11-20/hardware-feedback/probe-31-aba-refreshed.json) · [33 号对照记录](../results/2026-11-20/hardware-feedback/probe-33-control.json) · [厂商反馈接口说明](https://wiki.xgorobot.com/kb/the-xgo-series-robot/cmqcb93y800118ffs7q9nytyc?model=mini2s)

### 网页点动与桌面试跑

狗端 `xgo_web_control.py` 在 `127.0.0.1:18107` 提供只绑定回环地址的点动页面，用 SSH 隧道从 Mac 访问：

```bash
ssh -N -L 18107:127.0.0.1:18107 xgo-mini2s
# 浏览器打开 http://127.0.0.1:18107
```

页面只有前、后、左、右四个固定方向，按住才走。指令速度分 50%、75%、100%、125% 四档，默认 100%，下发中锁档，改档要在松开后进行；上限取 125%，使后退 −0.275 m/s 与转向 0.3125 rad/s 不越出训练指令范围。心跳 80 ms 续租，松手 0.32 s 内停机，单次最长 2 s。倾角超过 12°、读回关节偏离参考点 10°、读寄存器连续三次无有效应答、推理输出非有限值都会立即停止并保留错误信息。POST 请求校验 `Origin` 与页面内一次性令牌。

此前的两次试跑都在约 50 帧时以 `Invalid UART feedback for ROLL` 终止。`_read_register` 改为每次最多重发三次（单次超时 30 ms）后，2026-11-20 17:20–17:35 在桌面上做了 29 次短试跑，四个方向轮流，共 1,552 帧，没有再次出现读失败，重试计数也为 0，因此原故障未能复现；重发只是兜底，主因尚未定位。空闲状态另外做过 5,000 次只读压力测试，同样没有失败。

试跑同时记录每帧指令与读回的差异。相同姿态起步的 8 次试跑里，指令幅度 7.0°–9.1°，读回幅度 2.9°–5.0°，单帧最大跟踪误差 8.4°–9.6°，轮询 `GET /api/status` 可读到当次的 `track_deg`。读回与指令幅度之比在 1/3 到 2/3 之间，比 31/33 号单点动测到的五分之一高，两次测的姿态与幅度不同，尚不能直接对比。串口链路仍未通过关节位置跟踪验收，实机闭环行走保持关闭。

### 地面行走模式

桌面试跑的 ±6° 目标包络和 2°/tick 斜率会把策略的步态削平。用同一份策略做开环回放，前向 0.22 m/s 时 thigh 目标峰峰 12–25°、calf 22–32°；而桌面记录里实际发出的目标只有 thigh 4°、calf 4–8°，长度和斜率都不够，在地面上就是四腿撑住不动。

`xgo_web_control.py --walk` 把目标放开到舵机实测限位，每周期 4°，单次最长 6 秒，倾角上限 25°。2026-11-20 17:40 起在地面复测：0.5× 前进 1 秒（33 帧）倾角 2–6°，1.0× 前进 1.5 秒（51 帧）倾角 2–6°，现场确认能往前走。第一次从蹲姿（thigh 35°）直接起步时，策略先把四条腿撑到 57° 左右的默认姿态，同时起振导致前倾到 28° 触发保护；先让它空转一个周期站好再按前进即可。

> 注意：`--walk` 只改了目标包络与斜率，观测通道（姿态差分、关节位置/速度）没有动。
