"""Register a reproducible low-traction Mini2S task in Luwu's training stack."""

import math
import sys

import torch

import src.tasks  # noqa: F401  (register the upstream tasks)
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
    register_mjlab_task,
)

BASE_TASK = "Mini2S-Walk-Flat"
TASK = "Mini2S-Walk-LowFriction"
FOOT_GEOMS = tuple(f"robot/{leg}_calf_collision" for leg in ("fl", "fr", "br", "bl"))
FRICTION_RANGE = (0.07, 0.80)


def traction_geom_ids(model):
    return [model.geom(name).id for name in ("terrain", *FOOT_GEOMS)]


@requires_model_fields("geom_friction")
def sample_traction(env, env_ids, friction_range):
    """Use one log-uniform tangential coefficient for the floor and four feet."""
    lo, hi = friction_range
    if not (0 < lo <= hi < math.inf):
        raise ValueError(f"Invalid friction range: {friction_range}")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = torch.tensor(traction_geom_ids(env.sim.mj_model), device=env.device)
    mu = torch.empty((len(env_ids), 1), device=env.device)
    mu.uniform_(math.log(lo), math.log(hi)).exp_()
    env.sim.model.geom_friction[env_ids[:, None], ids[None, :], 0] = mu


def set_traction(cfg, friction_range):
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("ground_friction", None)
    cfg.events["traction"] = EventTermCfg(
        func=sample_traction,
        mode="startup",
        params={"friction_range": friction_range},
    )


def low_friction_env(play=False):
    cfg = load_env_cfg(BASE_TASK, play=play)
    set_traction(cfg, FRICTION_RANGE)

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.30, 0.35)
    twist.ranges.lin_vel_y = (-0.16, 0.16)
    twist.ranges.ang_vel_z = (-0.45, 0.45)
    if not play:
        twist.rel_standing_envs = 0.10
        # Train within a modest disturbance envelope on the most slippery worlds.
        push = cfg.events["push_robot"].params["velocity_range"]
        push.update(x=(-0.16, 0.16), y=(-0.16, 0.16), z=(-0.05, 0.05),
                    roll=(-0.20, 0.20), pitch=(-0.20, 0.20), yaw=(-0.25, 0.25))
    cfg.rewards["foot_slip"].weight = -0.25
    return cfg


def register_task():
    agent = load_rl_cfg(BASE_TASK)
    agent.experiment_name = "mini2s_velocity_walk_low_traction"
    register_mjlab_task(
        task_id=TASK,
        env_cfg=low_friction_env(),
        play_env_cfg=low_friction_env(play=True),
        rl_cfg=agent,
        runner_cls=load_runner_cls(BASE_TASK),
    )


def main():
    from scripts import play, train

    commands = {"train": train.main, "play": play.main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"Usage: {sys.argv[0]} {{train,play}} [Luwu options]")
    command = commands[sys.argv[1]]
    sys.argv[1] = TASK
    register_task()
    command()


if __name__ == "__main__":
    # Use the same import path when reloading the saved configuration.
    from xgo_low_friction import main as run

    run()
