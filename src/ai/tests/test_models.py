import pytest
import torch

from ...game.settings import Settings, easy_settings
from ..featurizer import featurize
from ..hyperparams import Hyperparams
from ..models import get_models
from ..rollout import do_rollout


@pytest.fixture
def hp():
    return Hyperparams()


@pytest.fixture
def settings():
    return easy_settings()


@pytest.fixture
def models(hp: Hyperparams, settings: Settings):
    return get_models(hp, settings)


@pytest.fixture
def rollout(settings: Settings):
    return do_rollout(settings)


@pytest.fixture
def inps(rollout, settings):
    rollouts = [rollout] * 5
    return featurize(
        [x["public_history"] for x in rollouts],
        [x["private_inputs"] for x in rollouts],
        [x["valid_actions"] for x in rollouts],
        settings,
        non_feature_dims=2,
    )


@pytest.fixture
def one_step_inp(rollout, settings):
    return featurize(
        rollout["public_history"][0],
        rollout["private_inputs"][0],
        rollout["valid_actions"][0],
        settings,
        non_feature_dims=0,
    )


def test_history_model_forward(hp, models, inps, one_step_inp):
    hist_model = models["hist"]
    hist_model.eval()

    output = hist_model(inps["hist"], seq_lengths=inps["seq_lengths"])
    assert len(output.shape) == 3 and output.shape[-1] == hp.hist_output_dim

    # Test inference mode (no batch)
    hist_model.reset_state()
    assert hist_model.state is None

    output = hist_model(one_step_inp["hist"], seq_lengths=one_step_inp["seq_lengths"])
    assert output.shape == (hp.hist_output_dim,)
    assert hist_model.state is not None
    state1 = hist_model.state

    # Test state persistence
    output = hist_model(one_step_inp["hist"], seq_lengths=one_step_inp["seq_lengths"])
    assert output.shape == (hp.hist_output_dim,)
    assert hist_model.state is not None
    assert not torch.allclose(state1[0], hist_model.state[0])

    hist_model.reset_state()
    assert hist_model.state is None


def test_backbone_model_forward(hp, models, inps):
    backbone_model = models["backbone"]
    backbone_model.eval()

    output = backbone_model(
        inps["hist"], inps["private"], seq_lengths=inps["seq_lengths"]
    )
    assert len(output.shape) == 3 and output.shape[-1] == hp.backbone_output_dim


def test_policy_model_forward(hp, models, settings, inps):
    policy_model = models["policy"]
    policy_model.eval()

    inps["valid_actions"][:, :, -1] = -1  # Make last action invalid

    log_probs = policy_model(
        inps,
    )
    probs = torch.exp(log_probs)
    assert len(log_probs.shape) == 3
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0))
    assert torch.allclose(probs[..., -1], torch.tensor(0.0))


def test_value_head_forward(hp, models, settings, inps):
    backbone_model = models["backbone"]
    backbone_model.eval()
    value_head = models["value_head"]
    value_head.eval()

    output = backbone_model(
        inps["hist"], inps["private"], seq_lengths=inps["seq_lengths"]
    )
    value = value_head(output)
    assert len(value.shape) == 2
