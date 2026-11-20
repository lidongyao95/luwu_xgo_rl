"""Check the Mini2S hardware observation and action profile on 4522."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv


# 脚本按"放到 luwu_mjlab-main 根目录"的方式编写；从仓库子目录直接运行时补上 train/。
import sys as _sys
from pathlib import Path as _Path
_TRAIN = _Path(__file__).resolve().parent.parent / "train"
if _TRAIN.is_dir():
    _sys.path.insert(0, str(_TRAIN))

from xgo_hardware_profile import LEG_LIMITS_DEG, body_rate, hardware_env


def main():
    prev = torch.deg2rad(torch.tensor([[0.0, 0.0, 179.0]]))
    now = torch.deg2rad(torch.tensor([[0.0, 0.0, -179.0]]))
    dt = torch.tensor([0.025])
    rate = body_rate(now, prev, dt)
    torch.testing.assert_close(rate, torch.tensor([[0.0, 0.0, math.radians(2) / 0.025]]))

    cfg = hardware_env(play=True, sample_steps=(3, 3), yaw_deadband_deg=(0.2, 0.2))
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=42)
        assert obs["actor"].shape == (4, 56) and obs["critic"].shape == (4, 83)
        torch.testing.assert_close(obs["actor"][:, :3], torch.zeros_like(obs["actor"][:, :3]))
        assert env.step_dt == 0.025
        first = obs["actor"][:, :6].clone()
        robot = env.scene["robot"]
        target = env.action_manager.get_term("joint_pos")
        default = robot.data.default_joint_pos
        for step in range(1, 4):
            obs, *_ = env.step(torch.full((4, 15), 15.0, device=env.device))
            if step < 3:
                torch.testing.assert_close(obs["actor"][:, :6], first)
        assert not torch.equal(obs["actor"][:, :6], first)
        assert torch.isfinite(obs["actor"]).all()
        torch.testing.assert_close(obs["actor"][:, 23:26], torch.zeros_like(obs["actor"][:, 23:26]))
        torch.testing.assert_close(obs["actor"][:, 38:41], torch.zeros_like(obs["actor"][:, 38:41]))
        torch.testing.assert_close(target._processed_actions[:, 12:], default[:, 12:])
        assert (target._processed_actions[:, :12] - default[:, :12]).abs().max() <= math.radians(12) + 1e-6
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert torch.count_nonzero(env.obs_buf["actor"][0, :3]) == 0
        limits = torch.deg2rad(torch.tensor(LEG_LIMITS_DEG * 4, device=env.device))
        for sign, bound in ((1.0, limits[:, 1]), (-1.0, limits[:, 0])):
            raw = torch.full((4, 15), sign * 15, device=env.device)
            for _ in range(60):
                target.process_actions(raw)
            torch.testing.assert_close(target.raw_action, raw)
            torch.testing.assert_close(target._processed_actions[:, :12], bound.expand(4, -1))
            torch.testing.assert_close(target._processed_actions[:, 12:], default[:, 12:])
    finally:
        env.close()
    print("Hardware profile checks passed")


if __name__ == "__main__":
    main()
