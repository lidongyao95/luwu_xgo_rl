"""Mini2S training profile for the posture-angle UART feedback on this robot."""

from dataclasses import dataclass, replace
import math
import sys

import torch
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls, register_mjlab_task
from mjlab.utils.lab_api.math import euler_xyz_from_quat


# 脚本按"放到 luwu_mjlab-main 根目录"的方式编写；从仓库子目录直接运行时补上 train/。
import sys as _sys
from pathlib import Path as _Path
_TRAIN = _Path(__file__).resolve().parent.parent / "train"
if _TRAIN.is_dir():
    _sys.path.insert(0, str(_TRAIN))

from xgo_low_friction import BASE_TASK, low_friction_env

TASK = "Mini2S-Walk-HardwareFeedback"
CONTROL_DT = 0.025
FILTER_ALPHA = 0.12
SLEW_RAD = math.radians(4.0)
# Bounds of the installed xgolib MINI servo encoding, with the runner's ±90° cap.
LEG_LIMITS_DEG = ((-73.0, 57.0), (-66.0, 90.0), (-31.0, 31.0))


def wrap_angle(angle):
    return (angle + math.pi).remainder(2 * math.pi) - math.pi


def body_rate(now, prev, dt):
    dr, dp, dy = (wrap_angle(now - prev) / dt[:, None]).unbind(dim=1)
    roll, pitch = now[:, 0], now[:, 1]
    sr, cr = roll.sin(), roll.cos()
    sp, cp = pitch.sin(), pitch.cos()
    return torch.stack((dr - dy * sp, dp * cr + dy * sr * cp, -dp * sr + dy * cr * cp), dim=1)


class SerialAttitude(ManagerTermBase):
    """Sample and hold read_roll/pitch/yaw, then emulate the dog's derivative filter."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg
        self.last_step = torch.full((self.num_envs,), -1, device=self.device)
        self.period = torch.empty_like(self.last_step)
        self.yaw_deadband = torch.empty(self.num_envs, device=self.device)
        self.angles = torch.zeros((self.num_envs, 3), device=self.device)
        self.rate = torch.zeros_like(self.angles)
        env.xgo_serial_attitude = self
        self.reset(None)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -1
        self.rate[ids] = 0
        lo, hi = self.cfg.params["sample_steps"]
        self.period[ids] = torch.randint(lo, hi + 1, self.period[ids].shape, device=self.device)
        lo, hi = self.cfg.params["yaw_deadband_deg"]
        radians = torch.empty_like(self.yaw_deadband[ids]).uniform_(math.radians(lo), math.radians(hi))
        self.yaw_deadband[ids] = radians

    def __call__(self, env, sample_steps, yaw_deadband_deg):
        del sample_steps, yaw_deadband_deg
        step = env.common_step_counter
        elapsed = step - self.last_step
        first = self.last_step < 0
        sample = first | (elapsed >= self.period)
        angles = torch.stack(euler_xyz_from_quat(env.scene["robot"].data.root_link_quat_w), dim=1)
        angle_noise = math.radians(0.02) * torch.randn_like(angles[:, :2])
        angles[:, :2] += angle_noise
        quantum = math.radians(0.01)
        angles = wrap_angle((angles / quantum).round() * quantum)
        yaw_delta = wrap_angle(angles[:, 2] - self.angles[:, 2])
        keep_yaw = ~first & (yaw_delta.abs() < self.yaw_deadband)
        angles[:, 2] = torch.where(keep_yaw, self.angles[:, 2], angles[:, 2])

        dt = elapsed.clamp_min(1) * env.step_dt
        raw = body_rate(angles, self.angles, dt)
        alpha = 1.0 - (1.0 - FILTER_ALPHA) ** (dt[:, None] / CONTROL_DT)
        filtered = torch.lerp(self.rate, raw, alpha)
        self.rate.copy_(torch.where((sample & ~first)[:, None], filtered, self.rate))
        self.angles.copy_(torch.where(sample[:, None], angles, self.angles))
        self.last_step.copy_(torch.where(sample, step, self.last_step))
        return self.rate


def serial_gravity(env):
    # The actor evaluates base_ang_vel first, so both terms use the same sample.
    roll, pitch = env.xgo_serial_attitude.angles[:, :2].unbind(dim=1)
    return torch.stack((pitch.sin(), -roll.sin() * pitch.cos(), -roll.cos() * pitch.cos()), dim=1)


def leg_feedback(env, velocity):
    values = joint_vel_rel(env) if velocity else joint_pos_rel(env)
    values[:, 12:15] = 0.0
    return values


class LimitedLegTargets(JointPositionAction):
    """Keep the 15-output ABI while holding the arm and slew-limiting leg targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if len(self.target_names) != 15 or any(
            name.startswith("arm_") != (i >= 12) for i, name in enumerate(self.target_names)
        ):
            raise ValueError(f"Unexpected Mini2S action order: {self.target_names}")
        self._last_target = self.offset.clone()
        self._leg_limits = torch.deg2rad(torch.tensor(LEG_LIMITS_DEG * 4, device=self.device))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._last_target[env_ids] = self.offset[env_ids]

    def process_actions(self, actions):
        super().process_actions(actions)
        wanted = self._processed_actions[:, :12].clamp(self._leg_limits[:, 0], self._leg_limits[:, 1])
        prev = self._last_target[:, :12]
        prev.add_((wanted - prev).clamp(-SLEW_RAD, SLEW_RAD))
        self._processed_actions = self._last_target


@dataclass(kw_only=True)
class LimitedLegTargetsCfg(JointPositionActionCfg):
    def build(self, env):
        return LimitedLegTargets(self, env)


class HardwareRunner(load_runner_cls(BASE_TASK)):
    """Initialize adaptation from the baseline networks with a fresh optimizer."""

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True}
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


def hardware_env(play=False, sample_steps=(1, 3), yaw_deadband_deg=(0.05, 0.40)):
    cfg = low_friction_env(play=play)
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 5
    actor = cfg.observations["actor"].terms
    actor["base_ang_vel"] = replace(
        actor["base_ang_vel"], func=SerialAttitude, noise=None,
        params={"sample_steps": sample_steps, "yaw_deadband_deg": yaw_deadband_deg},
    )
    actor["projected_gravity"] = replace(
        actor["projected_gravity"], func=serial_gravity, noise=None, params={},
    )
    for name in ("joint_pos", "joint_vel"):
        actor[name] = replace(
            actor[name], func=leg_feedback, noise=None, params={"velocity": name == "joint_vel"},
        )
    cfg.actions["joint_pos"] = LimitedLegTargetsCfg(entity_name="robot", actuator_names=(".*",), scale=0.25)
    # The UART runner exposes forward and yaw commands and holds lateral command at zero.
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    return cfg


def main():
    from scripts import play, train

    commands = {"train": train.main, "play": play.main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"Usage: {sys.argv[0]} {{train,play}} [Luwu options]")
    command = commands[sys.argv[1]]
    sys.argv[1] = TASK
    agent = load_rl_cfg(BASE_TASK)
    agent.experiment_name = "mini2s_walk_hardware_feedback"
    agent.algorithm.learning_rate = 3.0e-4
    register_mjlab_task(
        task_id=TASK, env_cfg=hardware_env(), play_env_cfg=hardware_env(play=True),
        rl_cfg=agent, runner_cls=HardwareRunner,
    )
    command()


if __name__ == "__main__":
    from xgo_hardware_profile import main as run

    run()
