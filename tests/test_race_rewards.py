"""The race variants' reward and termination terms follow the Arena's rules, step for step."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import microduck_challenges

ROOT = Path(microduck_challenges.__file__).parent
# The race branches each carry rewards.py in one challenge folder only.
FOLDERS = [f for f in ("roller_sprint_2m", "sprint_2m") if (ROOT / f / "rewards.py").exists()]


def _rewards(folder):
    return importlib.import_module(f"microduck_challenges.{folder}.rewards")


class _Scene(dict):
    """env.scene["robot"] and env.scene.env_origins, all the terms read."""


def _env(num_envs=2):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            default_root_state=torch.zeros(num_envs, 13),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
        )
    )
    scene = _Scene(robot=robot)
    scene.env_origins = torch.zeros(num_envs, 3)
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.02,
        scene=scene,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        robot=robot,
    )


@pytest.mark.parametrize("folder", FOLDERS)
def test_progress_rewards_distance_not_velocity_samples(folder):
    race = _rewards(folder)
    env = _env(1)
    env.episode_length_buf[:] = 1
    total = 0.0
    for k, x in enumerate([0.01, 0.0, 0.03, 0.02, 0.05]):  # jitter ending 5 cm from the spawn
        env.robot.data.root_link_pos_w[0, 0] = x
        env.episode_length_buf[:] = k + 1
        total += float(race.race_progress(env)) * env.step_dt
    assert total == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("folder", FOLDERS)
def test_progress_restarts_with_the_episode(folder):
    race = _rewards(folder)
    env = _env(1)
    env.robot.data.root_link_pos_w[0, 0] = 1.0
    env.episode_length_buf[:] = 50
    race.race_progress(env)
    env.robot.data.root_link_pos_w[0, 0] = 0.0  # reset to the spawn
    env.episode_length_buf[:] = 1
    assert float(race.race_progress(env)) == pytest.approx(0.0)


@pytest.mark.parametrize("folder", FOLDERS)
def test_fall_needs_fifteen_consecutive_tilted_timed_steps(folder):
    race = _rewards(folder)
    env = _env(1)
    tilted = torch.tensor([[0.0, 0.0, 0.0]])  # projected gravity z = 0: on its side
    upright = torch.tensor([[0.0, 0.0, -1.0]])

    # A tilt through the 25 settle steps does not count.
    for k in range(1, 26):
        env.episode_length_buf[:] = k
        env.robot.data.projected_gravity_b = tilted
        assert not race.arena_fall(env).any()
    # 14 tilted timed steps: legal. One upright step resets the count.
    for k in range(26, 40):
        env.episode_length_buf[:] = k
        assert not race.arena_fall(env).any()
    env.episode_length_buf[:] = 40
    env.robot.data.projected_gravity_b = upright
    assert not race.arena_fall(env).any()
    # 15 consecutive: a fall, on the 15th.
    env.robot.data.projected_gravity_b = tilted
    for i, k in enumerate(range(41, 56)):
        env.episode_length_buf[:] = k
        fell = bool(race.arena_fall(env).any())
        assert fell == (i == 14)


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_line_is_two_metres_past_where_the_settle_ended(folder):
    race = _rewards(folder)
    env = _env(1)
    for k in range(1, 26):  # a flying start: 0.6 m covered in the settle
        env.robot.data.root_link_pos_w[0, 0] = 0.6 * k / 25
        env.episode_length_buf[:] = k
        assert not race.race_finished(env).any()
    env.robot.data.root_link_pos_w[0, 0] = 2.59
    env.episode_length_buf[:] = 60
    assert not race.race_finished(env).any()
    env.robot.data.root_link_pos_w[0, 0] = 2.61
    env.episode_length_buf[:] = 61
    assert race.race_finished(env).all()


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_clock_runs_after_the_settle_only(folder):
    race = _rewards(folder)
    env = _env(1)
    env.episode_length_buf[:] = 25
    assert float(race.time_penalty(env)) == 0.0
    env.episode_length_buf[:] = 26
    assert float(race.time_penalty(env)) == 1.0


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_bonus_is_paid_in_full_after_dt_scaling(folder):
    race = _rewards(folder)
    env = _env(1)
    env.termination_manager = SimpleNamespace(get_term=lambda name: torch.tensor([True]))
    assert float(race.finish_bonus(env)) * env.step_dt == pytest.approx(1.0)  # x the weight, 20


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_lane_is_free_to_0_3_m_either_side(folder):
    race = _rewards(folder)
    env = _env(3)
    env.robot.data.root_link_pos_w[:, 1] = torch.tensor([0.25, -0.3, -0.5])
    assert race.lane_excess(env).tolist() == pytest.approx([0.0, 0.0, 0.2])


@pytest.mark.parametrize("folder", FOLDERS)
def test_the_race_variant_keeps_the_arenas_observation_layout(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    assert "race_state" in cfg.observations["critic"].terms and "race_state" not in actor
    assert cfg.episode_length_s == 4.0
    assert "push_robot" not in cfg.events
    assert cfg.terminations["fell_over"].func is _rewards(folder).arena_fall
    assert cfg.terminations["success"].func is _rewards(folder).race_finished
