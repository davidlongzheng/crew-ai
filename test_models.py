import pytest
import torch

from src.ai.hyperparams import Hyperparams
from src.ai.models import get_models
from src.game.settings import Settings


@pytest.fixture
def hp():
    return Hyperparams()


@pytest.fixture
def models(hp: Hyperparams):
    return get_models(hp, Settings())


def test_decision_model_forward_with_history(
    hp,
    models,
):
    decision_model = models["decision"]
    seq_len = 5

    # Test with history input
    hist_input = torch.randn(hp.batch_size, seq_len - 1, hp.embed_dim + 1)
    private_input = torch.randn(hp.batch_size, seq_len, hp.embed_dim)

    output = decision_model(hist_input, private_input)
    assert output.shape == (hp.batch_size, seq_len, hp.decision_output_dim)


def test_decision_model_forward_without_history(
    hp,
    models,
):
    decision_model = models["decision"]

    # Test without history input
    hist_input = None
    private_input = torch.randn(hp.embed_dim)

    output = decision_model(hist_input, private_input)
    assert output.shape == (hp.decision_output_dim,)


def test_policy_model_forward(
    hp,
    models,
):
    policy_model = models["policy"]
    num_actions = 3
    seq_len = 5

    hist_input = torch.randn(hp.batch_size, seq_len - 1, hp.embed_dim + 1)
    private_input = torch.randn(hp.batch_size, seq_len, hp.embed_dim)  # embed_dim
    valid_actions = torch.randn(hp.batch_size, seq_len, num_actions, hp.embed_dim)

    probs = policy_model(hist_input, private_input, valid_actions)
    assert probs.shape == (hp.batch_size, seq_len, num_actions)
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(hp.batch_size, seq_len)
    )  # probabilities sum to 1


def test_value_model_forward(
    hp,
    models,
):
    value_model = models["value"]
    seq_len = 5

    hist_input = torch.randn(hp.batch_size, seq_len - 1, hp.embed_dim + 1)
    private_input = torch.randn(hp.batch_size, seq_len, hp.embed_dim)  # embed_dim

    value = value_model(hist_input, private_input)
    assert value.shape == (hp.batch_size, seq_len)
