import pytest
from torchrl.envs import check_env_specs

from ...game.settings import easy_settings
from .. import environment


@pytest.fixture
def settings():
    return easy_settings()


@pytest.fixture
def env(settings):
    return environment.env(settings)


@pytest.fixture
def torchrl_env(settings):
    return environment.get_torchrl_env(settings)


def test_rollout(env, settings):
    env.reset(seed=42)
    num_iters = 0
    for agent in env.agent_iter():
        num_iters += 1
        observation, reward, termination, truncation, _ = env.last()
        assert not truncation

        if termination or truncation:
            assert reward > 0
            action = None
        else:
            assert reward == 0
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        env.step(action)
    env.close()
    assert num_iters == settings.num_players * (settings.num_tricks + 1)


def test_torchrl_env(torchrl_env):
    rollout = torchrl_env.rollout(max_steps=1000)
    assert rollout.batch_size == (18,)
    check_env_specs(torchrl_env)
