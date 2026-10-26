"""Race-reward terms: train for the Arena's clock instead of for a commanded speed.

The Arena times a policy from the end of a 0.5 s untimed settle (25 control steps) to the moment the
trunk has moved 2.0 m along world +x, and calls it a fall only after 15 consecutive control steps
with the trunk tilted past 60 degrees (projected gravity z above -0.5). These terms copy those rules
exactly, so a policy can use what the rules allow: speed carried into the start line, and a finish
dive that tips the trunk for fewer than 15 steps.

All rewards are rates (per second): the reward manager multiplies each by the control step (0.02 s).
`finish_bonus` therefore returns 1/dt on the finishing step, so its weight is the bonus itself.
"""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

SETTLE_STEPS = 25  # the Arena's 0.5 s untimed settle, in 50 Hz control steps
FALL_HOLD_STEPS = 15  # 0.3 s: consecutive tilted steps that make a fall
FALL_GRAVITY_Z = -0.5  # tilted: projected gravity z above this (more than 60 degrees)
FINISH_DISTANCE_M = 2.0


def _spawn_xy(env) -> torch.Tensor:
    """Where each env's duck was put at reset (the reset event has zero x/y range)."""
    asset = env.scene["robot"]
    return env.scene.env_origins[:, :2] + asset.data.default_root_state[:, :2]


def _xy_from_spawn(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, :2] - _spawn_xy(env)


def race_progress(env, v_clip: float = 4.0) -> torch.Tensor:
    """Forward speed along world +x (m/s), every control step including the settle.

    The displacement over the control step divided by dt, not the instantaneous velocity: the
    rewards then sum to the distance travelled, so a trunk that vibrates at the control rate
    cannot farm it. Clipped against physics spikes.
    """
    x = _xy_from_spawn(env)[:, 0]
    prev = getattr(env, "_race_prev_x", None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros_like(x)
    prev = torch.where(env.episode_length_buf <= 1, torch.zeros_like(prev), prev)
    env._race_prev_x = x.clone()
    return ((x - prev) / env.step_dt).clamp(-v_clip, v_clip)


def finish_bonus(env) -> torch.Tensor:
    """1/dt on the step the duck crosses the line (weight = the total bonus)."""
    return env.termination_manager.get_term("success").float() / env.step_dt


def fall_penalty(env) -> torch.Tensor:
    """1/dt on the step the Arena's fall rule ends the episode (use a negative weight = the cost).

    In the Arena a fall counts as the time limit (40-50 s). Without a cost here, the first thing
    a from-scratch policy finds is the dive: throw the trunk forward in the settle, lie tilted for
    the 15 steps it takes to be declared fallen, and keep the 0.9 m it dived (first attempt: every
    episode ended at exactly step 40).
    """
    return env.termination_manager.get_term("fell_over").float() / env.step_dt


def time_penalty(env, settle_steps: int = SETTLE_STEPS) -> torch.Tensor:
    """1 on every timed step (use a negative weight): the clock runs from the end of the settle."""
    return (env.episode_length_buf > settle_steps).float()


def lane_excess(env, half_width: float = 0.3) -> torch.Tensor:
    """Sideways distance (m) beyond half_width from the spawn line (the Arena's walls are at 0.4)."""
    y = _xy_from_spawn(env)[:, 1]
    return (y.abs() - half_width).clamp(min=0.0)


def race_state(env) -> torch.Tensor:
    """Critic-only: (x from spawn, y from spawn, seconds since reset). Lets the value function
    see how close the line is, which the actor is never told."""
    xy = _xy_from_spawn(env)
    t = (env.episode_length_buf.float() * env.step_dt).unsqueeze(-1)
    return torch.cat([xy, t], dim=-1)


# ── Terminations ─────────────────────────────────────────────────────────────────────────────


def arena_fall(
    env,
    settle_steps: int = SETTLE_STEPS,
    hold_steps: int = FALL_HOLD_STEPS,
    gravity_z: float = FALL_GRAVITY_Z,
) -> torch.Tensor:
    """The Arena's fall rule: `hold_steps` consecutive timed steps with projected gravity z > gravity_z.

    The Arena starts counting at the first timed step, so a tilt during the settle does not count.
    No immediate tilt termination: tipping for fewer than hold_steps steps stays legal.
    """
    gz = env.scene["robot"].data.projected_gravity_b[:, 2]
    tipped = (gz > gravity_z) & (env.episode_length_buf > settle_steps)
    count = getattr(env, "_race_tip", None)
    if count is None or count.shape[0] != env.num_envs:
        count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    count = torch.where(tipped, count + 1, torch.zeros_like(count))
    env._race_tip = count
    return count >= hold_steps


def race_finished(
    env, settle_steps: int = SETTLE_STEPS, distance: float = FINISH_DISTANCE_M
) -> torch.Tensor:
    """Success: the trunk is `distance` m past where it was when the settle ended."""
    x = env.scene["robot"].data.root_link_pos_w[:, 0]
    x0 = getattr(env, "_race_x0", None)
    if x0 is None or x0.shape[0] != env.num_envs:
        x0 = x.clone()
    x0 = torch.where(env.episode_length_buf == settle_steps, x, x0)
    env._race_x0 = x0
    return (env.episode_length_buf > settle_steps) & ((x - x0) >= distance)


# ── Style terms the runs add (not part of the race reward itself) ─────────────────────────────


def wheel_spin(env, wheel_radius: float = 0.0175, vel_scale: float = 1.0) -> torch.Tensor:
    """The challenge's wheel_speed term without the command factor: tanh of the mean forward
    wheel speed (m/s equivalent). The race reward ignores the command, so this must too."""
    asset = env.scene["robot"]
    ids = []
    for pattern in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids.append(asset.find_joints(pattern)[0][0])
    omega = asset.data.joint_vel[:, ids].mean(dim=1)
    return torch.tanh(omega.clamp(min=0.0) * wheel_radius / vel_scale)


def flight(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    """1 when neither foot (wheels included) touches the ground."""
    contact_time = env.scene[sensor_name].data.current_contact_time
    return ((contact_time > 0.0).sum(dim=1) == 0).float()


def leg_inphase_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mean |q_left + q_right| over hip_pitch, knee and ankle (rad); 0 when the legs move as one.

    The left and right joints of this robot have mirrored axes (checked on the model: setting
    left +a and right -a moves both feet identically), so the legs are in phase when the two
    angles sum to a constant, which the mirrored default pose makes 0.
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_race_leg_ids") or env._race_leg_ids[0].device != env.device:
        left = [asset.find_joints([f"left_{b}"])[0][0] for b in ("hip_pitch", "knee", "ankle")]
        right = [asset.find_joints([f"right_{b}"])[0][0] for b in ("hip_pitch", "knee", "ankle")]
        env._race_leg_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    left, right = env._race_leg_ids
    q = asset.data.joint_pos
    return (q[:, left] + q[:, right]).abs().mean(dim=-1)


def trunk_low(env, min_z: float = 0.10) -> torch.Tensor:
    """How far (m) the trunk sits below min_z above the floor: a sled, not a skater (use a negative weight)."""
    z = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (min_z - z).clamp(min=0.0)
