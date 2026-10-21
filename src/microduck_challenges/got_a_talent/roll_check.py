"""Does a MicroDuck Got a Talent policy roll on its timeline's triggers, on the Arena's stage?

Plays the stage the way the Arena scores an entry (arena.evaluate, the star alone on its own
keyframes, as check.py does), then reads the star's trace one trigger pulse at a time (twist[1]
going to 1), from the pulse to the next one or the end:

- rolled: the trunk turned ROLLED_DEG or more BACKWARD about its own lateral axis;
- stood: after that, standing (arena.stage.standing: trunk above 0.10 m, within 20 deg of upright)
  for STAND_HOLD_S straight.

Then, over the whole performance, how far the trunk travelled backward from where it started and
how far it drifted sideways of that line. The stage's fall rule is not reported: every roll
holds the trunk past 60 deg for longer than 0.3 s. Prints a line per pulse and exits 1 unless
every pulse rolled and stood.

    uv run --group workshop python -m microduck_challenges.got_a_talent.roll_check <policy.onnx> [--timeline <file>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from arena import stage
from arena.cast import cast_of
from arena.constants import CTRL_DT
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.timeline import Timeline, TimelineError, load_troupe, solo

EVENT = "got-a-talent"
TIMELINE = Path(__file__).with_name("timeline.json")
ROLLED_DEG = 330.0
STAND_HOLD_S = 0.5


def backward_rotation(trace: np.ndarray) -> np.ndarray:
    """Cumulative backward rotation (rad) of the trunk about its own lateral axis, per frame.

    Each step's relative rotation, conj(q0) * q1, is in the trunk's own frame; its y component,
    negated, is the backward part (the trunk faces +x, nose-up is negative y).
    """
    q = trace[:, 3:7].astype(np.float64)
    w0, v0 = q[:-1, 0], -q[:-1, 1:]
    w1, v1 = q[1:, 0], q[1:, 1:]
    w = w0 * w1 - np.sum(v0 * v1, axis=1)
    v = w0[:, None] * v1 + w1[:, None] * v0 + np.cross(v0, v1)
    flip = w < 0
    w[flip], v[flip] = -w[flip], -v[flip]
    norm = np.linalg.norm(v, axis=1)
    angle = 2 * np.arctan2(norm, w)
    axis_y = np.divide(v[:, 1], norm, out=np.zeros_like(norm), where=norm > 1e-12)
    return np.concatenate([[0.0], np.cumsum(-angle * axis_y)])


def _pulses(timeline: Timeline) -> list[float]:
    """Times the roll trigger goes to 1."""
    starts, previous = [], 0.0
    for t, command in zip(timeline.times, timeline.commands):
        if command[1] >= 0.5 > previous:
            starts.append(t)
        previous = command[1]
    return starts


def _travel(trace: np.ndarray) -> tuple[float, float]:
    """(metres travelled backward, metres drifted sideways) from the start, in the start's heading."""
    w, x, y, z = trace[0, 3:7].astype(np.float64)
    forward = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    moved = trace[-1, 0:2].astype(np.float64) - trace[0, 0:2]
    return float(-moved @ forward), float(abs(moved @ left))


def report(trace: np.ndarray, timeline: Timeline) -> tuple[list[str], bool, int]:
    """A line per trigger pulse then the travel line, whether every pulse rolled and stood, and
    how many did."""
    rotation = np.degrees(backward_rotation(trace))
    standing = stage.standing(trace)
    hold = round(STAND_HOLD_S / CTRL_DT)
    pulses = _pulses(timeline)
    lines, good = [], 0
    for k, t in enumerate(pulses):
        a = round(t / CTRL_DT)
        b = round(pulses[k + 1] / CTRL_DT) if k + 1 < len(pulses) else len(trace)
        gained = rotation[a:b] - rotation[a]
        turned = float(gained.max()) if b > a else 0.0
        line = f"t={t:5.2f}  roll: {turned:4.0f} deg"
        if turned < ROLLED_DEG:
            lines.append(line + f" (not round, {ROLLED_DEG:.0f} needed)")
            continue
        rolled_at = a + int(np.argmax(gained >= ROLLED_DEG))
        run, stood_at = 0, None
        for i in range(rolled_at, b):
            run = run + 1 if standing[i] else 0
            if run >= hold:
                stood_at = (i + 1 - hold) * CTRL_DT
                break
        if stood_at is None:
            lines.append(line + f", round at {rolled_at * CTRL_DT:.2f} s, never stood {STAND_HOLD_S} s")
            continue
        good += 1
        lines.append(line + f", round at {rolled_at * CTRL_DT:.2f} s, standing at {stood_at:.2f} s")
    back, side = _travel(trace)
    lines.append(f"travelled {back:.2f} m backward, drifted {side:.2f} m sideways")
    return lines, bool(pulses) and good == len(pulses), good


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a MicroDuck Got a Talent policy's backward rolls on the Arena's stage.")
    parser.add_argument("onnx")
    parser.add_argument("--timeline", default=str(TIMELINE))
    args = parser.parse_args(argv)
    event = load_event(EVENT)
    try:
        timeline = load_troupe(Path(args.timeline), event, allow_local=True).ducks[0].timeline
        policy = load_policy(Path(args.onnx))
        result = evaluate(event, policy, cast_of(event, solo(timeline), policy, None, {}).performers)
    except (TimelineError, PolicyError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    lines, passed, good = report(result.trace[0], timeline)
    passed = passed and result.seeds_finished == 1
    print("\n".join(lines))
    print(f"{'PASS' if passed else 'FAIL'}: {good}/{len(_pulses(timeline))} rolls rolled and stood")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
