"""MicroDuck Got a Talent trains in the Arena's physics, and reads the command a timeline carries."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

import microduck_challenges  # noqa: F401  (registers every task)
from arena.assets import ASSET_ROOT
from arena.constants import CTRL_DT, DECIMATION, DEFAULT_POSE, JOINT_NAMES, TRACE_WIDTH
from arena.course import compile_course
from arena.manifest import load_event
from arena.timeline import Timeline, load_troupe
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module("microduck_challenges.got_a_talent.env").make_env_cfg()


@pytest.fixture(scope="module")
def arena():
    return compile_course(load_event("got-a-talent"))


def _training_model(entity):
    """The robot as training compiles it: its MJCF with the entity's collision
    settings applied (mjlab's Entity runs each CollisionCfg.edit_spec on the spec)."""
    spec = entity.spec_fn()
    for collision in entity.collisions:
        collision.edit_spec(spec)
    return spec.compile()


def _body(model, geom):
    return model.body(int(model.geom_bodyid[geom])).name


def _robot_geoms(model, skip):
    return [g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g]) and _body(model, g) not in skip]


def _colliding(model, skip):
    """(body, geom type) of every geom that can collide, bodies in `skip` left out."""
    return sorted((_body(model, g), int(model.geom_type[g])) for g in _robot_geoms(model, skip))


# mjlab's flat terrain (terrain_entity._import_ground_plane) is a plane with
# MuJoCo's default contact: priority 0, condim 3, sliding friction 1.
FLOOR = (0, 3, 1.0)


def _floor_contacts(model, skip):
    """(body, condim, sliding friction) of each robot geom's contact with that floor.
    MuJoCo pairs two geoms by priority first; at equal priority it takes the larger
    condim and the larger friction."""
    out = []
    for g in _robot_geoms(model, skip):
        priority, condim, friction = int(model.geom_priority[g]), int(model.geom_condim[g]), float(model.geom_friction[g][0])
        if priority > FLOOR[0]:
            pair = (condim, friction)
        elif priority < FLOOR[0]:
            pair = FLOOR[1:]
        else:
            pair = (max(condim, FLOOR[1]), max(friction, FLOOR[2]))
        out.append((_body(model, g), *pair))
    return sorted(out)


def test_got_a_talent_collides_like_the_arena(cfg, arena):
    # A sit puts the trunk and hips on the floor: training must collide with the
    # same robot geoms as the Arena's stage, or a seated trunk sinks in one and
    # not the other.
    robot = _training_model(cfg.scene.entities["robot"])
    assert _colliding(robot, skip={"world"}) == _colliding(arena, skip={"world", "course"})


def test_got_a_talent_touches_the_floor_like_the_arena(cfg, arena):
    """Every robot geom meets the floor with the Arena's condim and friction.
    Robot-on-robot contacts are where the two differ: training gives its non-foot
    geoms condim 1 (FULL_COLLISION), so two of them touch without friction, and
    the Arena's are all condim 3."""
    floor = arena.geom("floor").id
    assert (int(arena.geom_priority[floor]), int(arena.geom_condim[floor]), float(arena.geom_friction[floor][0])) == FLOOR
    robot = _training_model(cfg.scene.entities["robot"])
    assert _floor_contacts(robot, skip={"world"}) == _floor_contacts(arena, skip={"world", "course"})


def test_got_a_talent_solves_contacts_like_the_arena(cfg, arena):
    assert cfg.sim.mujoco.iterations == arena.opt.iterations
    assert cfg.sim.mujoco.ls_iterations == arena.opt.ls_iterations
    assert math.isclose(cfg.sim.mujoco.timestep, arena.opt.timestep)
    assert cfg.decimation == DECIMATION


def test_got_a_talent_plays_on_after_a_fall_like_the_stage(cfg):
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_got_a_talent_reads_what_a_timeline_carries(cfg):
    """twist is [sit flag, roll trigger, 0], head_pose the 4 head deltas, body zeros: a
    keyframe's 13 numbers, in the 61-D observation's command slot."""
    from microduck_challenges.got_a_talent import roll

    assert isinstance(cfg.commands["twist"], roll.RollCommandCfg)
    assert cfg.commands["twist"].sit_prob == 0.0
    assert len(cfg.commands["head_pose"].ranges) == 4
    actor = cfg.observations["actor"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    assert actor["head_command"].params == {"command_name": "head_pose"}
    assert actor["body_command"].func is microduck_mdp.zero_command_padding
    assert actor["body_command"].params == {"dim": 6}


def test_checks_head_joint_order_matches_head_pose_trackings_own(cfg):
    """check.HEAD_JOINTS reads the head command's 4 dims as
    [neck_pitch, head_pitch, head_yaw, head_roll]; that must be the same order
    head_pose_tracking's own find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
    resolves them in, given this robot's actual joint declaration order."""
    from microduck_challenges.got_a_talent.check import HEAD_JOINTS

    entity = cfg.scene.entities["robot"].build()
    _, names = entity.find_joints_by_actuator_names(microduck_mdp._NECK_JOINT_PATTERNS)
    assert [JOINT_NAMES[i] for i in HEAD_JOINTS] == names


TIMELINE = Path(microduck_challenges.__file__).parent / "got_a_talent" / "timeline.json"


# Pollen's sit-and-stand performance, the one check.py was written for: the check tests play it.
SIT_TIMELINE = {"timeline_version": 1, "duration_s": 20.0, "keyframes": [
    {"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0]},
    {"t": 2.0, "twist": [1, 0, 0], "head": [0, 0, 0.5, 0]},
    {"t": 4.5, "twist": [1, 0, 0], "head": [0, 0, -0.5, 0]},
    {"t": 7.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0]},
    {"t": 9.0, "twist": [0, 0, 0], "head": [0, 0.4, 0, 0]},
    {"t": 11.0, "twist": [1, 0, 0], "head": [0, 0, 0, 0]},
    {"t": 15.0, "twist": [0, 0, 0], "head": [0, 0, 0.5, 0]},
    {"t": 17.0, "twist": [0, 0, 0], "head": [0, 0, -0.5, 0]},
    {"t": 19.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0]},
]}


@pytest.fixture
def sit_timeline(tmp_path):
    import json

    path = tmp_path / "sit.json"
    path.write_text(json.dumps(SIT_TIMELINE))
    return path


def test_the_reference_timeline_plays_on_the_arena():
    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    assert timeline.duration_s == 20.0
    assert len(timeline.times) == 13  # stand, then six roll pulses on and off


def test_the_reference_timeline_stays_inside_what_training_saw(cfg):
    """Roll pulses as training fires them: the sit flag 0, the trigger 0 or 1, each pulse as long
    as a trained one, pulses no closer than training's gap, the head within 40 % of its range."""
    from microduck_challenges.got_a_talent.env import ROLL_GAP_S, ROLL_PULSE_S
    from microduck_challenges.got_a_talent.roll_check import _pulses

    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    final = cfg.curriculum["head_pose_range"].params["range_stages"][-1]["ranges"]
    for command in timeline.commands:
        assert command[0] == 0.0 and command[1] in (0.0, 1.0) and command[2] == 0.0
        assert command[7:] == (0.0,) * 6
        for value, (_, hi) in zip(command[3:7], final):
            assert abs(value) <= 0.4 * hi
    starts = _pulses(timeline)
    assert starts
    for t in starts:
        end = next(u for u, c in zip(timeline.times, timeline.commands) if u > t and c[1] == 0.0)
        assert ROLL_PULSE_S[0] <= end - t <= ROLL_PULSE_S[1]
    assert all(b - a >= ROLL_GAP_S[0] for a, b in zip(starts, starts[1:]))


STAND = (0.0,) * 13
SIT_LEFT = (1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0) + (0.0,) * 6
EIGHT_S = Timeline(8.0, (0.0, 2.0, 5.0), (STAND, SIT_LEFT, STAND))


def _trace(z_at, yaw_at=lambda t: 0.0, tilt_at=lambda t: 0.0, duration_s=8.0):
    """A synthetic performance: trunk height, head yaw delta and roll tilt as functions of time."""
    n = round(duration_s / CTRL_DT)
    trace = np.zeros((n, TRACE_WIDTH), dtype=np.float32)
    for i in range(n):
        t = i * CTRL_DT
        half = np.radians(tilt_at(t)) / 2
        trace[i, 2] = z_at(t)
        trace[i, 3:7] = (np.cos(half), np.sin(half), 0.0, 0.0)
        trace[i, 7:] = DEFAULT_POSE
        trace[i, 7 + JOINT_NAMES.index("head_yaw")] += yaw_at(t)
    return trace


def _seated_between(a, b):
    return lambda t: 0.06 if a <= t < b else 0.115


def _report(trace):
    from microduck_challenges.got_a_talent import check

    return check.report(trace, EIGHT_S, load_event("got-a-talent").termination)


def test_a_performance_that_follows_its_timeline_passes():
    lines, passed, head = _report(_trace(_seated_between(3.0, 5.8), yaw_at=lambda t: 0.5 if 2.3 <= t < 5.0 else 0.0))
    assert passed, lines
    assert len(lines) == 4  # three keyframes, then the fall line
    assert head < 0.01


def test_a_sit_never_reached_fails():
    lines, passed, _ = _report(_trace(lambda t: 0.115))
    assert not passed
    assert "sit: not reached within 1.5 s" in lines[1]


def test_a_sit_reached_late_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.8, 5.8)))
    assert not passed
    assert "sit: not reached within 1.5 s" in lines[1]


def test_a_sit_left_early_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.0, 4.0)))
    assert not passed
    assert "left at 4.00 s" in lines[1]


def test_a_head_far_from_its_command_fails():
    lines, passed, head = _report(_trace(_seated_between(3.0, 5.8), yaw_at=lambda t: 0.9 if 2.3 <= t < 5.0 else 0.0))
    assert not passed
    assert head == pytest.approx(0.4, abs=1e-3)
    assert "head: off by 0.40 rad" in lines[1]


def test_a_fall_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.0, 5.8), tilt_at=lambda t: 80.0 if t >= 6.0 else 0.0))
    assert not passed
    assert lines[-1] == "fell at 6.30 s"


def test_check_plays_the_stage_and_fails_a_walker(capsys, sit_timeline):
    """End to end on the Arena: the reference walker does not sit."""
    from microduck_challenges.got_a_talent import check

    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx"), "--timeline", str(sit_timeline)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 + 2  # a line per keyframe, the fall line, the verdict
    assert out[-1].startswith("FAIL: largest head error ")


def test_check_reads_a_troupes_star_alone(tmp_path, capsys):
    """A troupe's other ducks are other policies: the check plays the star alone, on its own keyframes."""
    import json

    from microduck_challenges.got_a_talent import check

    reference = SIT_TIMELINE
    troupe = tmp_path / "timeline.json"
    troupe.write_text(json.dumps({"timeline_version": 2, "duration_s": reference["duration_s"], "ducks": [
        {"policy": "self", "keyframes": reference["keyframes"]},
        {"policy": "local:got_a_talent/first@1", "start": {"x": 0.0, "y": 0.6, "yaw": 0}, "keyframes": [{"t": 0.0}]},
    ]}))
    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx"), "--timeline", str(troupe)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 + 2  # the star's keyframes, the fall line, the verdict


def test_check_refuses_a_malformed_timeline(tmp_path, capsys):
    """main() must fail cleanly, like the Arena's own CLI, instead of raising."""
    from microduck_challenges.got_a_talent import check

    bad = tmp_path / "timeline.json"
    bad.write_text("{nope")
    assert check.main(["policy.onnx", "--timeline", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "timeline.json" in err and "not a JSON file" in err


def test_the_manifest_says_what_the_policy_reads():
    """publish writes this into every got-a-talent manifest: the sit flag, the roll trigger and the head pose."""
    from mjlab_microduck import challenge as ch

    import microduck_challenges  # noqa: F401 — registers every challenge's task

    command = ch.for_task("Mjlab-GotATalent-MicroDuck").command
    assert command["twist"].startswith("[sit flag")
    assert "roll trigger" in command["twist"]
    assert "head_yaw" in command["head"]
