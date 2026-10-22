"""MicroDuck Got a Talent's backward roll: the trigger a timeline sends, the roll it starts, and
what training pays for it.

The command (twist) is [0, roll trigger, 0]. The trigger is a short pulse: 1 for well under a
roll's length, then 0. A duck STANDING while the trigger is 1 starts a backward roll; once rolling
it finishes the roll and stands up whatever the trigger says. The policy can tell all three cases
from its own state (upright and still, or tumbling), so it never has to count: the timeline keeps
the time, one pulse per roll (AGENTS.md: a clock in the policy's joints washes out under PPO).

Training keeps per-env roll state on the command term (not observed by the actor; the critic sees
it through ``roll_state_obs``):

- ``rolling``: set when a trigger meets a ready (upright, standing-height) duck; cleared once the
  roll is complete AND the duck has stood upright for ``stand_hold_s``, or after ``timeout_s``.
- ``accum`` / ``frontier`` / ``paid``: backward rotation of the trunk about its own lateral axis
  (rad), counted only while some robot geom touches the floor and the roll stays sagittal: the
  forward-roll task's run-1 lesson (microduck_rl roulade), where uncounted ballistic and side
  rolls were what the policy found first.

The rewards follow the forward-roll task's recipe (mjlab_microduck tasks/mdp.py, "Roulade"):
one potential-based progress signal (a full roll pays 2*pi worth whatever the path), a stand-up
annuity gated on roll completion rather than on a clock, and the challenge's own rest stack
(sit/stand posture, head tracking, stillness, speed caps) paused while a roll is under way, since
every one of those terms says "do not tumble". Mid-roll spawns (reverse curriculum) are the
forward-roll task's ``reset_roulade_state`` pitched backward, synced into this state.

Backward is NEGATIVE rotation about the trunk's +y: the forward-roll code's convention is
face-down = +y (``_ROULADE_FWD_SIGN``), and flip_feasibility.py measured the same on the Arena.

Symmetry augmentation must stay off (env.py ENABLE_SYMMETRY): its mirror negates twist[1], which
would turn the trigger into -1 in half the samples.
"""

import math
from dataclasses import dataclass

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp

from microduck_challenges.got_a_talent import roll_reference

BACKWARD = -1.0  # sign of a backward roll's rate about the trunk's +y

# Sagittal-flatness credit on the rotation (as the forward roll): full while the trunk's lateral
# axis is within 30 deg of horizontal, none past 60 deg (a roll over the shoulder).
_FLAT_FULL = 0.5
_FLAT_ZERO = 0.866


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = torch.clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z of the trunk's lateral (y) axis: 0 through any pure pitch, +-1 on a shoulder."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _up_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z of the trunk's up (z) axis: cos(tilt)."""
    return 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)


class RollCommand(microduck_mdp.SitStandCommand):
    """twist = [0, roll trigger, 0], with the roll state the rewards read.

    Resampling (``resampling_time_range``) spaces the trigger pulses; each resample starts a pulse
    of ``pulse_s`` with probability ``roll_prob`` (the rest train standing idle between rolls).
    The sit flag stays 0 (``sit_prob`` 0): the parent's standing target is what the rest stack and
    the stand-up after a roll aim for.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.pulse_left = torch.zeros(n, device=dev)
        self.rolling = torch.zeros(n, dtype=torch.bool, device=dev)
        self.accum = torch.zeros(n, device=dev)
        self.frontier = torch.zeros(n, device=dev)
        self.paid = torch.zeros(n, device=dev)
        self.roll_time = torch.zeros(n, device=dev)
        self.stand_time = torch.zeros(n, device=dev)
        # Time since this roll started on roll_reference's clock (s); inf when there is none.
        self.ref_time = torch.full((n,), math.inf, device=dev)
        self.metrics["rolls_started"] = torch.zeros(n, device=dev)
        self.metrics["rolls_completed"] = torch.zeros(n, device=dev)

    def completion(self) -> torch.Tensor:
        """0 until the roll is ``gate_lo`` round, 1 past ``gate_hi``; 0 when not rolling."""
        return _smoothstep(self.frontier, self.cfg.gate_lo, self.cfg.gate_hi) * self.rolling.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = self.cfg.pulse_s
        length = torch.rand(n, device=self.device) * (hi - lo) + lo
        fire = torch.rand(n, device=self.device) < self.cfg.roll_prob
        self.pulse_left[env_ids] = torch.where(fire, length, torch.zeros_like(length))

    def compute(self, dt: float) -> None:
        super().compute(dt)
        env, cfg = self._env_ref, self.cfg
        self.pulse_left = torch.clamp(self.pulse_left - dt, min=0.0)
        trigger = self.pulse_left > 0
        self.vel_command_b[:, 1] = trigger.float()

        data = self.robot.data
        quat = data.root_link_quat_w
        z = torch.nan_to_num(data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
        up = torch.nan_to_num(_up_z(quat), nan=0.0)
        upright = up > math.cos(math.radians(cfg.stand_tilt_deg))

        # A trigger meets a duck ready to roll: a new roll starts from zero.
        start = trigger & ~self.rolling & upright & (z > cfg.ready_z)
        if start.any():
            for state in (self.accum, self.frontier, self.paid, self.roll_time, self.stand_time):
                state[start] = 0.0
            self.ref_time[start] = 0.0
            self.rolling |= start
            self.metrics["rolls_started"] += start.float()

        # Backward rotation, counted while supported and sagittal.
        rate = BACKWARD * torch.nan_to_num(data.root_link_ang_vel_b[:, 1], nan=0.0)
        found = env.scene.sensors[cfg.support_sensor].data.found
        supported = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()
        flat = _smoothstep(_FLAT_ZERO - torch.nan_to_num(_lateral_axis_z(quat), nan=1.0).abs(),
                           0.0, _FLAT_ZERO - _FLAT_FULL)
        rolling = self.rolling.float()
        self.accum += rate * dt * supported * flat * rolling
        self.frontier = torch.maximum(self.frontier, self.accum)
        self.roll_time += dt * rolling

        # Done: round (then, unless end_when_round, standing upright for stand_hold_s). Or out of time.
        standing = upright & (z > cfg.stand_z_min)
        self.stand_time = torch.where(standing & self.rolling, self.stand_time + dt, torch.zeros_like(self.stand_time))
        stood = (self.stand_time >= cfg.stand_hold_s) | cfg.end_when_round
        done = self.rolling & (self.frontier >= cfg.gate_hi) & stood
        self.metrics["rolls_completed"] += done.float()
        self.rolling &= ~(done | (self.roll_time > cfg.timeout_s))
        self.ref_time += dt


@dataclass(kw_only=True)
class RollCommandCfg(microduck_mdp.SitStandCommandCfg):
    class_type: type = RollCommand
    sit_prob: float = 0.0
    # Probability a resample fires a trigger pulse, and the pulse's length (s).
    roll_prob: float = 0.7
    pulse_s: tuple[float, float] = (0.4, 1.0)
    # Ready to roll: upright within stand_tilt_deg and trunk above ready_z (m). Loose on purpose: a
    # tight gate let a leaning duck refuse rolls (roll1-9). The stand after a roll is the rest
    # stack's job; stand_hold_s still reads this tilt when end_when_round is off.
    stand_tilt_deg: float = 45.0
    ready_z: float = 0.09
    # Completion gate on the rotation frontier (rad): opens from gate_lo to gate_hi.
    gate_lo: float = math.radians(260.0)
    gate_hi: float = math.radians(330.0)
    # A roll ends standing above stand_z_min (m) for stand_hold_s, or after timeout_s.
    stand_z_min: float = 0.10
    stand_hold_s: float = 0.5
    timeout_s: float = 4.0
    # End the roll as soon as it is round, handing the stand-up to the rest stack. Waiting for the
    # stand kept roll4 "rolling" until the timeout, through the next trigger: the rest stack, which
    # pays the upright stand, was paused nearly all the time and it waited leaning 28 deg back.
    end_when_round: bool = True
    support_sensor: str = "robot_ground_contact"

    def build(self, env) -> "RollCommand":
        return RollCommand(self, env)


def _roll(env, command_name: str) -> RollCommand:
    return env.command_manager.get_term(command_name)


# ── Rewards ──────────────────────────────────────────────────────────────────


def roll_progress(env, command_name: str = "twist", max_rate: float = 8.0) -> torch.Tensor:
    """Paid increments of the backward-rotation frontier, as a rate (rad/s), capped at max_rate.

    Potential-based: a full roll pays 2*pi (times the weight) however it gets there, and
    rocking back and forth pays nothing twice. Rotation faster than max_rate is forfeited, not
    deferred, so a violent whip earns no more than a fast roll (the forward roll's run-1 lesson;
    the scripted roll on the Arena turned at about 7 rad/s). Nothing past one turn pays: roll3,
    uncapped, learned to tumble 560-720 deg per trigger and never stand.
    """
    term = _roll(env, command_name)
    frontier = torch.clamp(term.frontier, max=2 * math.pi)
    owed = torch.clamp(frontier - term.paid, min=0.0)
    pay = torch.clamp(owed, max=max_rate * env.step_dt)
    term.paid = torch.maximum(term.paid, frontier)
    return pay / env.step_dt * term.rolling.float()


def roll_landing(env, command_name: str = "twist", **composite) -> torch.Tensor:
    """The standing goal score (posture_composite against the stand target), paid only once the
    roll is round: "do nothing" and "stay standing at the trigger" never collect it."""
    return _roll(env, command_name).completion() * microduck_mdp.posture_composite(
        env, command_name=command_name, **composite)


def roll_rise(env, command_name: str = "twist", **rise) -> torch.Tensor:
    """Upward trunk motion once the roll is round: gradient out of the seated/supine landing."""
    return _roll(env, command_name).completion() * microduck_mdp.posture_rise_bootstrap(
        env, command_name=command_name, **rise)


def roll_imitate(env, command_name: str = "twist", joint_std: float = 0.5,
                 rotation_std: float = 1.0) -> torch.Tensor:
    """How closely the duck follows the scripted roll (roll_reference) since its roll started:
    the joints (rms error against joint_std, rad) times the rotation so far (against rotation_std,
    rad). Loose on purpose: the training physics turns the same joint targets less far than the
    Arena's stage did. 0 outside the reference's 1.6 s, so it shapes the roll and the feet-under
    landing and leaves the stand to the rest stack.
    """
    term = _roll(env, command_name)
    joints, rotation = _reference(env.device)
    active = term.ref_time < (len(rotation) - 0.5) * roll_reference.DT
    frame = torch.clamp(torch.nan_to_num(term.ref_time / roll_reference.DT, posinf=0.0).round().long(),
                        0, len(rotation) - 1)
    q = env.scene["robot"].data.joint_pos
    joint_err = ((q - joints[frame]) ** 2).mean(dim=-1)
    rot_err = (term.accum - rotation[frame]) ** 2
    score = torch.exp(-joint_err / joint_std**2) * torch.exp(-rot_err / rotation_std**2)
    return score * active.float()


HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def head_hold(env, command_name: str = "twist", head_command: str = "head_pose",
              grace_s: float = 1.0) -> torch.Tensor:
    """Squared distance (rad^2, summed) of the neck and head from their commanded pose (positive;
    negative weight), everywhere but a roll's first grace_s, where the head throw belongs.

    Quadratic on purpose: roll1-8 learned to fold the neck to its stop (~1.9 rad off), where the
    Gaussian head terms (head_pose_tracking, posture_composite's head factor) are flat and pull
    nothing; the trunk then leaned ~30 deg back to balance the head, failing the stage's 20 deg.
    """
    term = _roll(env, command_name)
    asset = env.scene["robot"]
    if not hasattr(env, "_roll_head_ids"):
        ids, _ = asset.find_joints(HEAD_JOINTS, preserve_order=True)
        env._roll_head_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    ids = env._roll_head_ids
    target = asset.data.default_joint_pos[:, ids] + env.command_manager.get_command(head_command)
    err = ((asset.data.joint_pos[:, ids] - target) ** 2).sum(dim=-1)
    throwing = term.rolling & (term.ref_time < grace_s)
    return err * (~throwing).float()


_REFERENCE: dict = {}


def _reference(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(joints [frames, 14], rotation [frames]) of roll_reference, on device."""
    if device not in _REFERENCE:
        _REFERENCE[device] = (torch.tensor(roll_reference.JOINTS, dtype=torch.float32, device=device),
                              torch.tensor(roll_reference.ROTATION, dtype=torch.float32, device=device))
    return _REFERENCE[device]


def reference_time_at(rotation: torch.Tensor) -> torch.Tensor:
    """roll_reference's time (s) at which the script first turned ``rotation`` (rad) backward."""
    _, ref = _reference(rotation.device)
    frame = torch.searchsorted(torch.cummax(ref, dim=0).values, rotation.contiguous())
    return torch.clamp(frame, max=len(ref) - 1).float() * roll_reference.DT


def roll_lateral(env, command_name: str = "twist", **params) -> torch.Tensor:
    """Sideways trunk velocity squared during a roll (positive; negative weight): keeps the line
    straight, so a string of rolls crosses the stage instead of wandering."""
    return _roll(env, command_name).rolling.float() * microduck_mdp.roulade_lateral_velocity_penalty(env, **params)


def roll_flatness(env, command_name: str = "twist", **params) -> torch.Tensor:
    """(lateral axis world-z) squared during a roll (positive; negative weight): steers back
    toward a sagittal roll before the flatness gate stops counting it."""
    return _roll(env, command_name).rolling.float() * microduck_mdp.roulade_flatness_penalty(env, **params)


def when_resting(env, term, memo: str, roll_command: str = "twist", **params) -> torch.Tensor:
    """Any rest-stack term, its value frozen while a roll is under way.

    The sit/stand stack pays upright, still, at height, slow: a roll breaks every one of them, so
    left on it would make the roll net-negative and never learned. Paused to 0 (roll1-9), it still
    made every roll cost the stack's whole income, and the policy learned to refuse rolls by
    waiting outside the start gate's tilt (32 deg in roll1, 60 in roll9fresh). Frozen at its value
    the step before the roll began, a roll costs nothing and standing well before a trigger pays
    through the roll too. ``term`` is the original reward function, ``memo`` a key for its frozen
    value, and ``params`` its own (its own ``command_name`` among them, hence ``roll_command``).
    """
    rolling = _roll(env, roll_command).rolling
    memos = env.__dict__.setdefault("_rest_memo", {})
    value = term(env, **params)
    if memo not in memos:
        memos[memo] = torch.zeros_like(value)
    frozen = memos[memo]
    frozen[~rolling] = value[~rolling]
    return torch.where(rolling, frozen, value)


# ── Reset and observation ────────────────────────────────────────────────────


def sync_roll_state(env, env_ids: torch.Tensor, command_name: str = "twist") -> None:
    """After ``reset_roulade_state`` (pitched backward): start each mid-roll spawn's roll at its
    spawn angle, and every standing spawn idle.

    ``reset_roulade_state`` leaves each spawn's pitch in env._roulade_max (0 standing, negative =
    backward mid-roll). Must run after it: events run in the order env.py adds them.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    term = _roll(env, command_name)
    spawn = torch.clamp(BACKWARD * env._roulade_max[env_ids], min=0.0)
    mid = spawn > 0
    term.rolling[env_ids] = mid
    for state in (term.accum, term.frontier, term.paid):
        state[env_ids] = spawn
    term.roll_time[env_ids] = 0.0
    term.stand_time[env_ids] = 0.0
    term.ref_time[env_ids] = torch.where(mid, reference_time_at(spawn), torch.full_like(spawn, math.inf))
    # A mid-roll spawn has no rest value from before its roll: it earns the rest stack by finishing.
    for frozen in env.__dict__.get("_rest_memo", {}).values():
        frozen[env_ids] = 0.0


def roll_state_obs(env, command_name: str = "twist") -> torch.Tensor:
    """Critic only: [rolling, frontier in turns, completion]. The actor infers it from its body."""
    term = _roll(env, command_name)
    return torch.stack([term.rolling.float(), term.frontier / (2 * math.pi), term.completion()], dim=-1)
