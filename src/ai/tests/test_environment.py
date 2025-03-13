import pytest
from torchrl.envs import check_env_specs

from ...game.settings import easy_settings
from ..environment import CrewEnv, get_torchrl_env


@pytest.fixture
def settings():
    return easy_settings()


@pytest.fixture
def env(settings):
    return CrewEnv(settings)


@pytest.fixture
def torchrl_env(settings):
    return get_torchrl_env(settings=settings, randomize_invalid_actions=True)


def test_rollout(env, settings):
    obs, _ = env.reset(seed=42)

    num_steps = 0
    while True:
        action = env.action_space.sample(obs["action_mask"])
        obs, reward, terminated, truncated, _ = env.step(action)
        num_steps += 1
        assert not truncated

        if terminated:
            assert reward > 0
            break
        else:
            assert reward == 0
    assert num_steps == settings.num_players * settings.num_tricks


def test_torchrl_env(torchrl_env):
    rollout = torchrl_env.rollout(max_steps=1000)
    assert rollout.batch_size == (18,)
    check_env_specs(torchrl_env)
