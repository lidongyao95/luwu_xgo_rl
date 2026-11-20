"""Compare Mini2S checkpoints on a fixed-friction, headless MuJoCo Warp matrix."""

import argparse
from dataclasses import asdict
from importlib.util import find_spec
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


# 脚本按"放到 luwu_mjlab-main 根目录"的方式编写；从仓库子目录直接运行时补上 train/。
import sys as _sys
from pathlib import Path as _Path
_TRAIN = _Path(__file__).resolve().parent.parent / "train"
if _TRAIN.is_dir():
    _sys.path.insert(0, str(_TRAIN))

from xgo_low_friction import BASE_TASK, set_traction, traction_geom_ids

COMMANDS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward_015", (0.15, 0.0, 0.0)),
    ("forward_025", (0.25, 0.0, 0.0)),
    ("turn_035", (0.0, 0.0, 0.35)),
)
DEFAULT_MU = (0.08, 0.12, 0.20, 0.35, 0.70)
TILT_ABORT_DEG = 25.0


def tilt_limit(env):
    """Compare simulated roll and pitch against the dog's host-side stop limit."""
    gx, gy, gz = env.scene["robot"].data.projected_gravity_b.unbind(dim=1)
    roll = torch.atan2(-gy, -gz)
    pitch = gx.clamp(-1, 1).asin()
    return torch.maximum(roll.abs(), pitch.abs()) > math.radians(TILT_ABORT_DEG)


def audit_contacts(env, friction, ids):
    """Check actual solver contacts after settling from the default pose."""
    with torch.no_grad():
        env.reset()
        for _ in range(15):
            env.step(torch.zeros((env.num_envs, 15), device=env.device))
    data = env.sim.wp_data
    count = int(data.nacon.numpy()[0])
    geom = data.contact.geom.numpy()[:count]
    contact_type = data.contact.type.numpy()[:count]
    mask = (
        (geom == ids[0]).any(axis=1)
        & np.isin(geom, ids[1:]).any(axis=1)
        & ((contact_type & 1) != 0)
    )
    worlds = data.contact.worldid.numpy()[:count][mask]
    tangent = data.contact.friction.numpy()[:count][mask, :2]
    expected = friction.cpu().numpy()
    if set(worlds) != set(range(env.num_envs)):
        raise RuntimeError("At least one test world has no foot-ground solver contact")
    np.testing.assert_allclose(tangent, np.repeat(expected[worlds, None], 2, axis=1))
    return len(worlds)


def build_env(frictions, trials, seconds, seed, device, timing_hz, feedback_steps=None):
    if feedback_steps is None:
        cfg = load_env_cfg(BASE_TASK, play=True)
    else:
        from mjlab.managers import TerminationTermCfg
        from xgo_hardware_profile import hardware_env

        cfg = hardware_env(play=True, sample_steps=(feedback_steps,) * 2,
                           yaw_deadband_deg=(0.2, 0.2))
        cfg.terminations["uart_tilt_limit"] = TerminationTermCfg(func=tilt_limit)
    cfg.seed = seed
    cfg.scene.num_envs = len(frictions) * len(COMMANDS) * trials
    cfg.episode_length_s = seconds + 2.0
    cfg.auto_reset = False
    cfg.events.pop("encoder_bias", None)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.15, 0.15)
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.commands["twist"].resampling_time_range = (seconds + 2.0,) * 2
    if timing_hz is not None:
        # Hold the integration step fixed for the 40 Hz vs 50 Hz comparison.
        cfg.sim.mujoco.timestep = 0.005
        cfg.decimation = 200 // timing_hz
    set_traction(cfg, (frictions[0], frictions[0]))
    env = ManagerBasedRlEnv(cfg, device=device)

    values = torch.tensor(frictions, device=device).repeat_interleave(len(COMMANDS) * trials)
    ids = traction_geom_ids(env.sim.mj_model)
    env.sim.model.geom_friction[:, ids, 0] = values[:, None]
    expected = values[:, None].expand(-1, len(ids))
    torch.testing.assert_close(env.sim.model.geom_friction[:, ids, 0], expected)
    arm_id = env.sim.mj_model.geom("robot/arm_calf_collision").id
    arm_mu = env.sim.mj_model.geom_friction[arm_id, 0]
    torch.testing.assert_close(
        env.sim.model.geom_friction[:, arm_id, 0], torch.full_like(values, arm_mu)
    )
    commands = torch.tensor([vec for _, vec in COMMANDS], device=device)
    commands = commands.repeat_interleave(trials, dim=0).repeat(len(frictions), 1)
    return env, commands, audit_contacts(env, values, ids)


def pin_commands(env, commands):
    """Keep the command fixed, including immediately after a partial reset."""
    cmd = env.command_manager.get_term("twist")
    cmd.command.copy_(commands)
    cmd.is_standing_env.zero_()


@torch.inference_mode()
def rollout(env, agent_cfg, checkpoint, commands, frictions, trials, seconds, seed):
    runner_cls = load_runner_cls(BASE_TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=env.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)
    env.reset(seed=seed)
    pin_commands(env, commands)
    # Refresh the actor observation after replacing the commands sampled at reset.
    # The play configuration has no actor observation delay or history.
    env.obs_buf = env.observation_manager.compute()
    obs = wrapped.get_observations()

    robot = env.scene["robot"]
    sensor = env.scene["feet_ground_contact"]
    sites, _ = robot.find_sites(
        [name.removesuffix("_calf_collision") for name in sensor.primary_names],
        preserve_order=True,
    )
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    heading = robot.data.heading_w.clone()
    forward = torch.stack((heading.cos(), heading.sin()), dim=1)

    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elapsed_steps = torch.zeros(env.num_envs, device=env.device)
    velocity_sum = torch.zeros_like(commands)
    velocity_error_sum = torch.zeros_like(commands)
    sample_count = torch.zeros_like(elapsed_steps)
    foot_slip_sum = torch.zeros_like(elapsed_steps)
    foot_contacts = torch.zeros_like(elapsed_steps)
    end_xy = start_xy.clone()
    parity_samples = []

    steps = round(seconds / env.step_dt)
    settle_steps = round(1.0 / env.step_dt)
    for step in range(steps):
        act = policy(obs)
        if step in (0, settle_steps, steps - 1):
            parity_samples.extend(zip(
                obs["actor"][::trials].cpu().numpy().copy(),
                act[::trials].cpu().numpy().copy(),
            ))
        if agent_cfg.clip_actions is not None:
            act = act.clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions)
        _, _, terminated, timeout, _ = env.step(act)
        velocity = torch.cat((
            robot.data.root_link_lin_vel_b[:, :2],
            robot.data.root_link_ang_vel_b[:, 2:3],
        ), dim=1)
        pos = robot.data.root_link_pos_w[:, :2]
        end_xy[active] = pos[active]
        elapsed_steps += active

        if step >= settle_steps:
            sample_count += active
            velocity_sum += torch.where(active[:, None], velocity, 0)
            error = (velocity - commands).abs()
            velocity_error_sum += torch.where(active[:, None], error, 0)
            contact = (sensor.data.found > 0) & active[:, None]
            slip = robot.data.site_lin_vel_w[:, sites, :2].norm(dim=-1)
            foot_slip_sum += torch.where(contact, slip, 0).sum(dim=1)
            foot_contacts += contact.sum(dim=1)

        # Reset terminated simulator worlds so the batch can advance. Their
        # metrics stop at the first end condition and retain the terminal pose.
        done = terminated | timeout
        active &= ~done
        if done.any():
            env.reset(env_ids=done.nonzero(as_tuple=True)[0])
            pin_commands(env, commands)
            env.obs_buf = env.observation_manager.compute()
        obs = wrapped.get_observations()

    mean_vel = velocity_sum / sample_count.clamp_min(1)[:, None]
    mean_error = velocity_error_sum / sample_count.clamp_min(1)[:, None]
    slip = foot_slip_sum / foot_contacts.clamp_min(1)
    displacement = end_xy - start_xy
    progress = (displacement * forward).sum(dim=1)
    drift = displacement.norm(dim=1)
    valid = sample_count > 0
    metrics = torch.cat((
        active[:, None], elapsed_steps[:, None] * env.step_dt, mean_vel,
        mean_error, slip[:, None], progress[:, None], drift[:, None],
    ), dim=1).cpu().numpy()
    valid = valid.cpu().numpy()
    contacts = foot_contacts.cpu().numpy()
    if not np.isfinite(metrics).all():
        raise RuntimeError(f"Nonfinite rollout metrics for {checkpoint}")

    results = []
    for group, (mu, (name, vec)) in enumerate(
        (mu, command) for mu in frictions for command in COMMANDS
    ):
        batch = slice(group * trials, (group + 1) * trials)
        values = metrics[batch]
        velocity_samples = values[valid[batch], 2:8]
        slip_samples = values[contacts[batch] > 0, 8]
        means = velocity_samples.mean(axis=0).tolist() if len(velocity_samples) else [None] * 6
        results.append({
            "mu": mu, "command": name, "target_vx_vy_wz": vec,
            "completed": int(values[:, 0].sum()), "trials": trials,
            "mean_alive_s": float(values[:, 1].mean()),
            "mean_vx_vy_wz": means[:3], "mae_vx_vy_wz": means[3:],
            "mean_contact_slip_m_s": float(slip_samples.mean()) if len(slip_samples) else None,
            "mean_forward_m": float(values[:, 9].mean()),
            "mean_drift_m": float(values[:, 10].mean()),
        })
    return results, parity_samples


def check_onnx(path, parity_samples):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        engine = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        input_name = engine.get_inputs()[0].name
        backend = f"ONNX Runtime {ort.__version__} CPUExecutionProvider"
    else:
        engine = ReferenceEvaluator(model)
        input_name = engine.input_names[0]
        backend = f"ONNX {onnx.__version__} ReferenceEvaluator"
    errors = []
    for obs, expected in parity_samples:
        (action,) = engine.run(None, {input_name: obs[None, :]})
        np.testing.assert_allclose(action[0], expected, atol=1e-4, rtol=1e-4, equal_nan=False)
        errors.append(float(np.max(np.abs(action[0] - expected))))
    return {
        "path": str(path), "backend": backend, "samples": len(errors),
        "max_abs_action_error": max(errors),
        "joint_names": {kv.key: kv.value for kv in model.metadata_props}["joint_names"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, help="Check ONNX export of the last checkpoint.")
    parser.add_argument("--friction", type=float, nargs="+", default=DEFAULT_MU)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-hz", type=int, choices=(40, 50),
                        help="Compare policy update rates at a fixed 5 ms physics step.")
    parser.add_argument("--hardware-feedback-steps", type=int, choices=(1, 2, 3),
                        help="Use 40 Hz leg limits and sample attitude every 1–3 control steps.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or not np.isfinite(args.seconds) or args.seconds < 2:
        parser.error("Require positive trials and at least two seconds per trial.")
    if any(not np.isfinite(mu) or mu <= 0 for mu in args.friction):
        parser.error("Friction values must be positive and finite.")
    if args.hardware_feedback_steps is not None and args.timing_hz is not None:
        parser.error("The hardware feedback profile fixes the simulated control rate at 40 Hz.")

    env, commands, contact_count = build_env(
        args.friction, args.trials, args.seconds, args.seed, args.device, args.timing_hz,
        args.hardware_feedback_steps,
    )
    report = {
        "seed": args.seed, "trial_seconds": args.seconds,
        "physics_dt_s": env.physics_dt, "control_dt_s": env.step_dt,
        "verified_solver_contacts": contact_count, "runs": [],
    }
    if args.hardware_feedback_steps is not None:
        report["hardware_feedback"] = {
            "attitude_sample_interval_s": env.step_dt * args.hardware_feedback_steps,
            "yaw_deadband_deg": 0.2,
            "roll_pitch_abort_deg": TILT_ABORT_DEG,
        }
    try:
        for checkpoint in args.checkpoint:
            results, parity_samples = rollout(
                env, load_rl_cfg(BASE_TASK), checkpoint, commands,
                args.friction, args.trials, args.seconds, args.seed,
            )
            report["runs"].append({"checkpoint": str(checkpoint), "results": results})
        if args.onnx is not None:
            report["onnx_parity"] = check_onnx(args.onnx, parity_samples)
    finally:
        env.close()
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
