"""The 2 m Roller Sprint's training environment: Microduck on roller skates.

A complete copy of microduck_rl's roller velocity task
(microduck_velocity_rollers_env_cfg.py), made the roller sprint's own: the two
rewards a novice tunes are read from challenge.toml ([params]). Everything here
is yours to change; the robot (MICRODUCK_WALK_ROLLERS_ROBOT_CFG) and the reward
functions (microduck_mdp) come from mjlab_microduck and are not edited here.

MIGRATED to mjlab 1.3.0 + canonical BAM (2026-07), matching the velocity env's
sim2real machinery, and updated for the NEW roller model:

  - `get_walk_rollers_spec` now loads `robot_groundcontact_rollers.xml`
    (it silently loaded the wheel-less standup model before): 14 actuated
    joints + 4 passive wheels (passive_{L,R}{F,R}wheel), two per blade,
    INTERSPERSED in the joint order (after each ankle) — everything resolves
    joints by NAME, never by index.
  - Legs run the canonical BAM actuator like every other variant (was a plain
    XML PD — an actuator-physics mismatch, and no joint-friction DR).
  - Obs migrated to the unified 61D layout (twist + zero-padded head/body
    command slots) so roller policies load through the runtime's
    --new-cmd-obs path. Symmetry OFF (SYMMETRY_CFG is hardcoded for the old
    51D layout).
  - DR/noise/delays matched to the velocity env's FIXED (non-accumulating,
    per-env-verified) versions. Wheel-bearing frictionloss DR was REMOVED: the
    Arena's roller model's wheels are frictionless, and training against added
    bearing friction breaks the policy once raced there.

RACE VARIANT (race-skate / race-roller): the policy is trained for the Arena's clock, not for a
commanded speed. The reward is the trunk's progress along world +x from the first control step (the
speed carried into the start line is free time), a bonus for crossing 2.0 m after the 0.5 s settle,
and a small time penalty; the episode ends on the finish or on the Arena's own fall rule (15
consecutive tilted control steps), and lasts at most 4 s. The stock style terms (pose, gait, posture,
command tracking) are gone, pushes are off, and the start is the Arena's (STAND, joints +-0.01 rad,
yaw +-2 deg). See rewards.py and oct25/RL_RACE.md. SKATE_SHAPING adds what makes it skate: a penalty
for frames with no wheel on the ground, and the stock wheel spin and glide terms.
"""

import math
from copy import deepcopy

# race-skate: penalise flight, keep wheel spin and glide. race-roller: the race reward alone.
SKATE_SHAPING = True

# Symmetry — OFF: SYMMETRY_CFG's obs permutation is hardcoded for the old 51D
# layout and breaks on the 61D obs (same situation as all other v1.5+ envs).
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to the velocity env) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # BAM friction budget per-env (legs)
ENABLE_ARMATURE_RANDOMIZATION        = True   # legs only — NOT the wheel bearings
ENABLE_VELOCITY_PUSHES               = False  # race: the Arena pushes nothing
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True   # obs-level per-env rotation
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the velocity env unless roller-specific) ───────────────
COM_RANDOMIZATION_RANGE          = 0.003   # ±3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.2, 0.2)  # roller-specific: gentler than walk ±0.3
ARENA_YAW_RAD                    = math.radians(2.0)  # arena/events/roller-sprint-2m.toml [start.perturbation]
ARENA_JOINT_RAD                  = 0.01
SPAWN_HEIGHT_M                   = 0.12  # arena.course.SPAWN_HEIGHT_M; the rollers pop up to 0.139 in 10 steps, as in training
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# The head assembly's bodies, for the head CoM randomization. The head-roll
# body is named bottom_head_shell in the walk model and jaw_soft in the
# 2026-07 roller model, hence the alternation. NOTE: bearing_roll is NOT a
# head body — in both models it is the right-hip-yaw link (child of
# trunk_base); it has always been listed here by mistake and is kept only to
# preserve existing DR behavior.
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck import challenge
from . import rewards as race_rewards
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """The roller sprint's training environment."""

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)

    # 2026-07 model: the roller_blade bodies were merged into the ankles (blade
    # mesh is now a visual geom on ankle_{l,r}_v1); the tires hang directly off
    # the ankles. Each ankle subtree's only collision geoms are its two tires,
    # so this keeps the old per-foot semantics: 2 slots, left first.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(ankle_l_v1|ankle_r_v1)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # NOTE: an env-side action clip was tried here to bound the target, but the
    # deployment pipeline (infer_policy.py) does NOT clip → the clip would only
    # exist in sim, a train/deploy mismatch. The over-command deterrent lives
    # policy-side instead (action_over_limit reward below), baked into the network
    # so it transfers with the ONNX.

    # === REWARDS ===
    # The race reward (rewards.py), and nothing of the stock style recipe (pose, gait, posture,
    # command tracking, foot swing): only upright (a knob, weight 0.1 here: the Arena's fall rule
    # is the judge of tilt), a tiny action_rate_l2, self-collision and heading_hold stay.
    keep = {"upright", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = knobs["upright_reward"]
    cfg.rewards["action_rate_l2"].weight = -0.01

    cfg.rewards["progress"] = RewardTermCfg(func=race_rewards.race_progress, weight=20.0)
    cfg.rewards["finish_bonus"] = RewardTermCfg(func=race_rewards.finish_bonus, weight=20.0)
    cfg.rewards["time_penalty"] = RewardTermCfg(func=race_rewards.time_penalty, weight=-2.0)
    # A fall counts as the time limit in the Arena: price it like the finish is paid.
    cfg.rewards["fall"] = RewardTermCfg(func=race_rewards.fall_penalty, weight=-20.0)
    # The walls are at +-0.4 m; the policy is never told where it is in the lane.
    cfg.rewards["lane"] = RewardTermCfg(func=race_rewards.lane_excess, weight=-40.0)
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=0.5,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )

    # The knob term stays so the knob stays live; race-roller sets its value to 0 in challenge.toml.
    cfg.rewards["wheel_speed"] = RewardTermCfg(
        func=race_rewards.wheel_spin,
        weight=knobs["speed_reward"],
    )
    if SKATE_SHAPING:
        # Skating, not running: wheels on the ground (the leaders are airborne 60% of the time).
        # An attempt-tax, so it ramps in (microduck_rl AGENTS.md: skill discovery first).
        cfg.rewards["flight"] = RewardTermCfg(
            func=race_rewards.flight,
            weight=-3.0,
            params={"sensor_name": "feet_ground_contact"},
        )
        cfg.curriculum["flight_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "flight",
                "weight_stages": [
                    {"step": 0, "weight": -3.0},
                    {"step": 250 * 24, "weight": -6.0},
                    {"step": 500 * 24, "weight": -10.0},
                ],
            },
        )
        # race-skate2 ended up sledding (trunk or head on the floor 36% of the time, 8% airborne).
        cfg.rewards["trunk_low"] = RewardTermCfg(func=race_rewards.trunk_low, weight=-100.0, params={"min_z": 0.10})
        # Push-and-glide: coast on one blade with quiet legs (the challenge's term and weight).
        cfg.rewards["glide"] = RewardTermCfg(
            func=microduck_mdp.glide_reward,
            weight=4.0,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "vel_ref": 0.2,
            },
        )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )
    # The Arena's fall rule, exactly (15 consecutive tilted control steps), in place of the stock
    # 70 degree tilt limit: tipping for fewer steps stays legal, so a finish dive can be learned.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=race_rewards.arena_fall, time_out=False
    )
    # The Arena's finish: 2.0 m past where the trunk was when the 0.5 s settle ended.
    cfg.terminations["success"] = TerminationTermCfg(
        func=race_rewards.race_finished, time_out=False
    )
    cfg.episode_length_s = 4.0

    # === EVENTS ===
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    del cfg.events["foot_friction"]  # wheels roll; ground friction lives in the XML

    cfg.events.pop("push_robot", None)  # the template's pushes; the Arena pushes nothing
    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # The Arena's start: STAND at the lane's origin, heading +x within +-2 degrees, every
    # actuated joint within +-0.01 rad of the stand pose. (The stock reset spreads the duck over
    # a 1 m square at any heading, which a race along world +x cannot use.)
    cfg.events["reset_base"].params["pose_range"].update(
        {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (SPAWN_HEIGHT_M, SPAWN_HEIGHT_M), "yaw": (-ARENA_YAW_RAD, ARENA_YAW_RAD)}
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (-ARENA_JOINT_RAD, ARENA_JOINT_RAD)
    cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",)
    )

    # No wheel-bearing friction DR: the Arena's roller model's wheels are
    # frictionless (frictionloss="0.0" in robot_allcollisions_rollers.xml's
    # passive_joint default). A 3000-iteration reference run that ramped bearing
    # frictionloss in from iteration 2000 stayed fast and consistent (2.99-3.22 s,
    # 5/5 seeds) through checkpoint 2000, then went erratic from checkpoint 2250
    # on (40 s 0/5, 17.7 s, 3.5 s, 24.2 s 3/5) once raced on the Arena's free wheels.

    # ── DR matched to the velocity env's FIXED versions ───────────────────────
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Legs/head only — the wheel bearings' tiny armature is excluded; the
        # Arena's wheels are frictionless and get no DR at all (see above).
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    # 1.3.0 base template adds sensor-based foot_height + height_scan; the roller
    # env has no terrain-height sensor.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    # IMU delay 0-1 control steps (matches velocity: the real dxl IMU path is fast)
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise — matched to the velocity env
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level, actor only — matches velocity)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity moving average)
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude the passive wheel joints from joint_pos/vel obs (obs dim 14, matches
    # the action space). Deepcopy per group so the encoder-bias `biased` flag
    # below applies to the actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Privileged: how far along the lane the duck is, and the clock. The actor is never told.
    cfg.observations["critic"].terms["race_state"] = ObservationTermCfg(func=race_rewards.race_state)

    # Privileged wheel speeds for the critic (4 wheels in the new model).
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # Command obs parity with the 61D family layout: head/body slots zero-padded
    # (the roller task drives heading through the twist slot instead).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False  # RelativeHeadingVelocityCommand handles heading internally
    command.ranges.heading = None    # must be None when heading_command=False
    # The Arena commands one constant vx from its sweep, 0.30-2.00 m/s, and keeps the best time.
    # The race reward ignores the command (a policy that ignores it is what the sweep needs), so
    # the stock 0.6 cap, which kept wheel_speed from out-weighing upright, does not apply: sample
    # the whole sweep. One command per episode, held for the whole run, as the Arena holds it.
    command.ranges.lin_vel_x = (0.3, 2.0)
    command.resampling_time_range = (100.0, 100.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z range is the clip limit for cmd[2] = heading error (rad).
    # Set to 0 → cmd[2] is always 0 → no turning demand (straight-line focus).
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.RelativeHeadingVelocityCommandCfg(**vars(command))

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # (no action_rate curriculum: the race keeps it tiny and constant, -0.01)

    # (heading_tracking_weight curriculum removed — heading is disabled while we
    # focus on straight-line skating. Re-add together with the reward above.)

    # CoM randomization curricula — velocity's ramp, capped lower for the
    # balance-sensitive skating task (audit lesson: ±30 mm forced a nervous
    # gait on the walker; skates are even less forgiving).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


RL_CFG = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches the family; normalizer baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,  # race: 0.03 only grew the action noise (entropy 20 -> 36) while every episode dived
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_sprint_2m",  # logs/rsl_rl/roller_sprint_2m/
    run_name="roller_sprint_2m",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
