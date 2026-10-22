"""MicroDuck Got a Talent's backward roll: the trigger, the roll state and its rewards in training,
and roll_check's reading of a performance."""

import importlib
import math

import numpy as np
import pytest
import torch

import microduck_challenges  # noqa: F401  (registers every task)
from arena.constants import CTRL_DT, DEFAULT_POSE, JOINT_NAMES, TRACE_WIDTH
from arena.timeline import Timeline
from microduck_challenges.got_a_talent import roll, roll_check

ENV = "microduck_challenges.got_a_talent.env"


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module(ENV).make_env_cfg()


# ── Configuration ────────────────────────────────────────────────────────────


def test_symmetry_stays_off_because_its_mirror_negates_the_trigger():
    assert importlib.import_module(ENV).ENABLE_SYMMETRY is False


def test_the_rest_stack_pauses_during_a_roll_and_the_regularisers_do_not(cfg):
    env = importlib.import_module(ENV)
    for name in env.REST_TERMS:
        assert cfg.rewards[name].func is roll.when_resting, name
        assert callable(cfg.rewards[name].params["term"])
    for name in ("action_rate_l2", "joint_torque_rate_l2", "self_collisions"):
        assert cfg.rewards[name].func is not roll.when_resting
    assert cfg.rewards["roll_progress"].weight > 0
    assert cfg.rewards["roll_lateral"].weight < 0 and cfg.rewards["roll_flatness"].weight < 0


def test_the_actor_reads_61_numbers_and_only_the_critic_sees_the_roll_state(cfg):
    assert "roll_state" in cfg.observations["critic"].terms
    assert "roll_state" not in cfg.observations["actor"].terms


def test_mid_roll_spawns_are_pitched_backward_and_synced_after(cfg):
    events = list(cfg.events)
    assert events.index("sync_roll_state") > events.index("set_ground_state")
    params = cfg.events["set_ground_state"].params
    assert params["midroll_pitch_min"] < params["midroll_pitch_max"] < 0


# ── roll_check on synthetic performances ─────────────────────────────────────


def _pitch_trace(pitch_at, z_at, duration_s=8.0, x_at=lambda t: 0.0):
    """Trunk pitched BACKWARD by pitch_at(t) degrees (nose up), at height z_at(t)."""
    n = round(duration_s / CTRL_DT)
    trace = np.zeros((n, TRACE_WIDTH), dtype=np.float32)
    for i in range(n):
        t = i * CTRL_DT
        half = math.radians(pitch_at(t)) / 2
        trace[i, 0] = x_at(t)
        trace[i, 2] = z_at(t)
        trace[i, 3:7] = (math.cos(half), 0.0, -math.sin(half), 0.0)  # backward = -y
        trace[i, 7:] = DEFAULT_POSE
    return trace


def _ramp(a, b, to):
    return lambda t: 0.0 if t < a else to if t >= b else to * (t - a) / (b - a)


ROLL_AT_2 = Timeline(8.0, (0.0, 2.0, 2.6), ((0.0,) * 13, (0.0, 1.0) + (0.0,) * 11, (0.0,) * 13))


def test_backward_rotation_counts_a_full_turn_backward():
    trace = _pitch_trace(_ramp(1.0, 2.0, 360.0), lambda t: 0.1, duration_s=3.0)
    assert np.degrees(roll_check.backward_rotation(trace))[-1] == pytest.approx(360.0, abs=1.0)
    forward = _pitch_trace(_ramp(1.0, 2.0, -90.0), lambda t: 0.1, duration_s=3.0)
    assert np.degrees(roll_check.backward_rotation(forward))[-1] == pytest.approx(-90.0, abs=1.0)


def test_a_roll_that_comes_round_and_stands_passes():
    lines, passed, good = roll_check.report(
        _pitch_trace(_ramp(2.2, 3.2, 360.0), lambda t: 0.05 if 2.2 <= t < 3.6 else 0.115,
                     x_at=_ramp(2.2, 3.2, -0.3)), ROLL_AT_2)
    assert passed and good == 1, lines
    assert "standing at 3.6" in lines[0]
    assert lines[-1].startswith("travelled 0.30 m backward, drifted 0.00 m")


def test_a_half_roll_fails():
    lines, passed, _ = roll_check.report(
        _pitch_trace(_ramp(2.2, 3.2, 180.0), lambda t: 0.05 if t >= 2.2 else 0.115), ROLL_AT_2)
    assert not passed
    assert "not round" in lines[0]


def test_a_roll_that_never_stands_fails():
    lines, passed, _ = roll_check.report(
        _pitch_trace(_ramp(2.2, 3.2, 360.0), lambda t: 0.05 if t >= 2.2 else 0.115), ROLL_AT_2)
    assert not passed
    assert "never stood" in lines[0]


# ── The training environment, end to end on CPU ──────────────────────────────


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = importlib.import_module(ENV).make_env_cfg()
    cfg.scene.num_envs = 8
    cfg.curriculum["roll_start"].params["start_step"] = 0  # rolls on from the first step
    environment = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield environment
    environment.close()


def _backward_lean():
    """Joint targets that throw the duck backward: hips forward, head back (the scripted roll's
    opening, flip_feasibility.py)."""
    q = np.zeros(len(JOINT_NAMES))
    for name, delta in (("left_hip_pitch", 0.92), ("right_hip_pitch", -0.92), ("left_knee", -1.14),
                        ("right_knee", 1.14), ("left_ankle", 0.88), ("right_ankle", -0.88),
                        ("neck_pitch", 0.5), ("head_pitch", -1.13)):
        q[JOINT_NAMES.index(name)] = delta
    return torch.tensor(np.tile(q, (8, 1)), dtype=torch.float32)


def test_mid_roll_spawns_start_rolling_at_their_spawn_angle(env):
    """A mid-roll spawn is rolling from its spawn angle; a standing spawn starts at 0 (rolling only
    if a trigger pulse met it at reset)."""
    env.reset()
    term = env.command_manager.get_term("twist")
    frontier = torch.rad2deg(term.frontier)
    mid = frontier > 1.0
    assert mid.any() and (~mid).any()
    assert term.rolling[mid].all()
    assert torch.all(frontier[mid] >= 59.0) and torch.all(frontier[mid] <= 301.0)
    assert torch.all(term.paid[mid] == term.frontier[mid])


def test_no_trigger_no_roll_and_a_trigger_starts_one_that_counts_backward(env):
    env.reset()
    term = env.command_manager.get_term("twist")
    standing = term.frontier < 1e-3  # standing spawns (mid-roll ones keep rolling)
    term.rolling &= ~standing  # cancel any roll a pulse at reset started
    term.cfg.roll_prob = 0.0
    term.pulse_left[:] = 0.0
    term.time_left[:] = 100.0  # no resample during the test
    hold = torch.zeros(8, len(JOINT_NAMES))
    for _ in range(5):
        env.step(hold)
    assert not term.rolling[standing].any()
    assert env.command_manager.get_command("twist")[:, 1].sum() == 0

    names = list(env.reward_manager.active_terms)
    pose = names.index("posture_composite")
    term.pulse_left[:] = 0.6
    paid = torch.zeros(8)
    held = []
    for _ in range(25):
        env.step(_backward_lean())
        paid += env.reward_manager._step_reward[:, names.index("roll_progress")]
        held.append(env.reward_manager._step_reward[:, pose].clone())
    assert term.rolling[standing].all()
    assert torch.all(torch.rad2deg(term.frontier[standing]) > 60.0)
    assert torch.all(paid[standing] > 0.0)
    # The rest stack holds its pre-roll value while rolling: a roll neither earns nor costs it.
    assert torch.equal(held[-1][standing], held[-5][standing])


def test_roll_progress_pays_one_turn_and_nothing_past_it():
    import types

    term = types.SimpleNamespace(frontier=torch.tensor([0.0, 0.0]), paid=torch.zeros(2),
                                 rolling=torch.tensor([True, True]))
    env = types.SimpleNamespace(step_dt=CTRL_DT)
    total = torch.zeros(2)
    for turned in torch.linspace(0.0, 3.0 * math.pi, 400):  # one and a half turns
        term.frontier = torch.stack([turned, torch.clamp(turned, max=math.pi)])
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(roll, "_roll", lambda env, name: term)
            total += roll.roll_progress(env) * CTRL_DT
    assert total[0].item() == pytest.approx(2 * math.pi, abs=1e-3)
    assert total[1].item() == pytest.approx(math.pi, abs=1e-3)


def test_a_round_roll_ends_at_once_and_hands_the_stand_to_the_rest_stack(env):
    env.reset()
    term = env.command_manager.get_term("twist")
    term.cfg.roll_prob = 0.0
    term.pulse_left[:] = 0.0
    term.time_left[:] = 100.0
    term.rolling[:] = True
    term.frontier[:] = term.cfg.gate_hi
    term.accum[:] = term.cfg.gate_hi
    term.stand_time[:] = 0.0
    env.step(torch.zeros(8, len(JOINT_NAMES)))
    assert not term.rolling.any()


def test_the_reference_starts_at_the_home_pose_and_goes_round_by_one_second(env):
    from microduck_challenges.got_a_talent import roll_reference as ref

    home = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
    assert np.abs(np.array(ref.JOINTS[0]) - home).max() < 0.1
    assert ref.ROTATION[round(1.0 / ref.DT)] >= math.radians(330.0)
    assert len(ref.JOINTS) == len(ref.ROTATION) and all(len(row) == len(JOINT_NAMES) for row in ref.JOINTS)


def test_roll_imitate_pays_following_the_script_and_nothing_without_a_roll(env):
    from microduck_challenges.got_a_talent import roll_reference as ref

    env.reset()
    term = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    term.ref_time[:] = math.inf
    assert torch.all(roll.roll_imitate(env) == 0.0)
    frame = 20
    term.ref_time[:] = frame * ref.DT
    rotation = torch.tensor(ref.ROTATION)
    # A reference whose every frame is env 0's current pose (joint_pos is read back from the sim).
    pose = robot.data.joint_pos[0].clone().expand(len(rotation), -1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(roll, "_reference", lambda device: (pose, rotation))
        term.accum[:] = rotation[frame]
        on = roll.roll_imitate(env)
        term.accum[:] = rotation[frame] + math.pi
        off = roll.roll_imitate(env)
    assert on[0] > 0.99 and torch.all(off < 0.01)


def test_mid_roll_spawns_join_the_reference_at_their_angle(env):
    env.reset()
    term = env.command_manager.get_term("twist")
    mid = term.frontier > 1e-3
    # A standing spawn has no reference clock, unless a trigger at reset just started its roll.
    standing = term.ref_time[~mid]
    assert torch.all(torch.isinf(standing) | (standing <= 2 * env.step_dt))
    expected = roll.reference_time_at(term.frontier[mid])
    assert torch.allclose(term.ref_time[mid], expected, atol=0.05)


def test_head_hold_prices_a_folded_neck_except_during_the_throw(env, cfg):
    env.reset()
    term = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    assert cfg.rewards["head_hold"].weight < 0
    assert cfg.rewards["head_pose_tracking"].func is not roll.when_resting
    ids, _ = robot.find_joints(roll.HEAD_JOINTS, preserve_order=True)
    home = robot.data.default_joint_pos.clone()
    folded = home.clone()
    folded[:, ids[0]] -= 1.9  # the neck at its stop, as roll1-8 held it
    head = torch.zeros(8, 4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(env.command_manager, "get_command", lambda name: head)
        for pose, expect in ((home, 0.0), (folded, 1.9**2)):
            mp.setattr(type(robot.data), "joint_pos", property(lambda self, p=pose: p))
            term.rolling[:] = False
            assert torch.allclose(roll.head_hold(env), torch.full((8,), expect), atol=1e-4)
        term.rolling[:] = True
        term.ref_time[:] = 0.3  # the head throw at a roll's start is free
        assert torch.all(roll.head_hold(env) == 0.0)


def test_a_leaning_duck_cannot_refuse_a_roll(env):
    """roll1-9 learned to wait leaning past the start gate's tilt so no roll ever started."""
    assert env.command_manager.get_term("twist").cfg.stand_tilt_deg >= 45.0


def test_rolls_wait_until_the_stand_is_learned(env):
    params = env.event_manager.get_term_cfg("set_ground_state").params
    term = env.command_manager.get_term("twist")
    start = 600 * 24
    saved = env.common_step_counter
    try:
        env.common_step_counter = 0
        roll.roll_curriculum(env, None, start_step=start)
        assert term.cfg.roll_prob == 0.0 and params["midroll_prob"] == 0.0 and params["standing_prob"] == 1.0
        env.common_step_counter = start
        roll.roll_curriculum(env, None, start_step=start)
        assert term.cfg.roll_prob > 0.0 and params["midroll_prob"] > 0.0
    finally:
        env.common_step_counter = saved


def test_the_roll_script_mirrors_the_legs_and_holds_each_keyframe():
    from microduck_challenges.got_a_talent import roll_expert as rx

    v = rx.load_variants("cpu")
    ends, offsets = rx.keyframes(v["params"], v["n_keyframes"])
    hip_l, hip_r = JOINT_NAMES.index("left_hip_pitch"), JOINT_NAMES.index("right_hip_pitch")
    assert torch.allclose(offsets[..., hip_l], -offsets[..., hip_r])
    assert torch.all(ends[:, 1:] > ends[:, :-1])
    first = rx.script_action(ends, offsets, torch.tensor([0]), torch.tensor([0.0]))
    last = rx.script_action(ends, offsets, torch.tensor([0]), torch.tensor([99.0]))
    assert torch.allclose(first[0], offsets[0, 0]) and torch.allclose(last[0], offsets[0, -1])
    assert torch.all((v["handover"] >= rx.HANDOVER_RANGE[0]) & (v["handover"] <= rx.HANDOVER_RANGE[1]))
