import pytest
import torch

from ...game.settings import Settings
from ..featurizer import featurize
from ..hyperparams import Hyperparams


@pytest.fixture
def hp():
    return Hyperparams()


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def private_inputs():
    return [
        [
            {
                "hand": [(1, 0), (2, 0)],  # 2 cards
                "player_idx": 0,
                "trick": 1,
                "turn": 0,
            },
            {
                "hand": [(3, 0)],  # 1 card
                "player_idx": 1,
                "trick": 1,
                "turn": 1,
            },
        ],
        [
            {
                "hand": [(1, 0), (2, 0)],  # 2 cards
                "player_idx": 0,
                "trick": 1,
                "turn": 0,
            },
        ],
    ]


@pytest.fixture
def public_history():
    return [
        [
            {},
            {
                "player_idx": 0,
                "trick": 1,
                "card": (1, 0),
                "turn": 0,
            },
        ],
        [{}],
    ]


@pytest.fixture
def valid_actions():
    return [
        [[(1, 0), (2, 0)], [(3, 0)]],
        [[(3, 0)]],
    ]


def test_featurize_basic(settings, public_history, private_inputs, valid_actions):
    """Test basic featurization with non_feature_dims=0."""
    inps = featurize(
        public_history=public_history[0][0],
        private_inputs=private_inputs[0][0],
        valid_actions=valid_actions[0][0],
        settings=settings,
        non_feature_dims=0,
    )

    assert inps["seq_lengths"] is None
    assert inps["hist"]["player_idxs"].shape == ()
    assert inps["hist"]["tricks"].shape == ()
    assert inps["hist"]["cards"].shape == (2,)
    assert inps["hist"]["turns"].shape == ()
    assert inps["private"]["hand"].shape == (settings.max_hand_size, 2)
    assert inps["private"]["player_idx"].shape == ()
    assert inps["private"]["trick"].shape == ()
    assert inps["private"]["turn"].shape == ()
    assert inps["valid_actions"].shape == (settings.max_hand_size, 2)


def test_featurize_with_history(
    settings, public_history, private_inputs, valid_actions
):
    """Test featurization with public history."""
    inps = featurize(
        public_history=public_history[0][1],
        private_inputs=private_inputs[0][1],
        valid_actions=valid_actions[0][1],
        settings=settings,
        non_feature_dims=0,
    )

    assert inps["seq_lengths"] is None
    assert inps["hist"]["player_idxs"].shape == ()
    assert inps["hist"]["tricks"].shape == ()
    assert inps["hist"]["cards"].shape == (2,)
    assert inps["hist"]["turns"].shape == ()
    assert inps["private"]["hand"].shape == (settings.max_hand_size, 2)
    assert inps["private"]["player_idx"].shape == ()
    assert inps["private"]["trick"].shape == ()
    assert inps["private"]["turn"].shape == ()
    assert inps["valid_actions"].shape == (settings.max_hand_size, 2)


def test_featurize_sequence(settings, public_history, private_inputs, valid_actions):
    """Test featurization with sequence inputs (non_feature_dims=1)."""
    inps = featurize(
        public_history=public_history[0],
        private_inputs=private_inputs[0],
        valid_actions=valid_actions[0],
        settings=settings,
        non_feature_dims=1,
    )

    assert inps["hist"]["player_idxs"].shape == (2,)
    assert inps["hist"]["tricks"].shape == (2,)
    assert inps["hist"]["cards"].shape == (2, 2)
    assert inps["hist"]["turns"].shape == (2,)
    assert inps["private"]["hand"].shape == (2, settings.max_hand_size, 2)
    assert inps["private"]["player_idx"].shape == (2,)
    assert inps["private"]["trick"].shape == (2,)
    assert inps["private"]["turn"].shape == (2,)
    assert inps["valid_actions"].shape == (2, settings.max_hand_size, 2)
    assert inps["seq_lengths"] is None


def test_featurize_batch_sequence(
    settings, public_history, private_inputs, valid_actions
):
    """Test featurization with batch of sequences (non_feature_dims=2)."""
    inps = featurize(
        public_history=public_history,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        settings=settings,
        non_feature_dims=2,
    )

    assert inps["hist"]["player_idxs"].shape == (2, 2)
    assert inps["hist"]["tricks"].shape == (2, 2)
    assert inps["hist"]["cards"].shape == (2, 2, 2)
    assert inps["hist"]["turns"].shape == (2, 2)
    assert inps["private"]["hand"].shape == (2, 2, settings.max_hand_size, 2)
    assert inps["private"]["player_idx"].shape == (2, 2)
    assert inps["private"]["trick"].shape == (2, 2)
    assert inps["private"]["turn"].shape == (2, 2)
    assert inps["valid_actions"].shape == (2, 2, settings.max_hand_size, 2)
    assert (inps["seq_lengths"] == torch.tensor([2, 1])).all()


def test_featurize_invalid_dims(settings):
    """Test that invalid non_feature_dims raises assertion error."""
    with pytest.raises(AssertionError):
        featurize(
            public_history=None,
            private_inputs={},
            valid_actions=[],
            settings=settings,
            non_feature_dims=3,  # Invalid value
        )
