import pytest
import torch

from ..featurizer import featurize


class MockEmbedModel(torch.nn.Module):
    def __init__(self, output_dim: int = 4, drop_dims: tuple[int, ...] | None = None):
        super().__init__()
        self.output_dim = output_dim
        self.drop_dims = drop_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.drop_dims:
            drop_dims = {x if x >= 0 else len(shape) + x for x in self.drop_dims}
            shape = [x for i, x in enumerate(shape) if i not in drop_dims]

        # Return tensor of same shape as input but with output_dim features
        return torch.ones(*shape, self.output_dim)


@pytest.fixture
def embed_models():
    return {
        "player": MockEmbedModel(),
        "trick": MockEmbedModel(),
        "card": MockEmbedModel(drop_dims=(-1,)),
        "hand": MockEmbedModel(drop_dims=(-1, -2)),
    }


def test_featurize_basic(embed_models):
    """Test basic featurization with non_feature_dims=0."""
    private_inputs = {
        "hand": [(1, 0), (2, 0), (3, 0)],  # Some cards in hand
        "player_idx": 0,
        "trick": 1,
    }
    valid_actions = [(1, 0), (2, 0)]  # Valid card plays

    hist_inp, private_inp, valid_actions_inp = featurize(
        embed_models,
        public_history=None,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        non_feature_dims=0,
    )

    assert hist_inp is None  # No history provided
    assert isinstance(private_inp, torch.Tensor)
    assert isinstance(valid_actions_inp, torch.Tensor)
    assert valid_actions_inp.dtype == torch.int8


def test_featurize_with_history(embed_models):
    """Test featurization with public history."""
    public_history = {
        "player_idx": 0,
        "trick": 1,
        "card": (1, 0),
        "last_in_trick": True,
    }
    private_inputs = {
        "hand": [(2, 0), (3, 0)],
        "player_idx": 1,
        "trick": 1,
    }
    valid_actions = [(2, 0)]

    hist_inp, private_inp, valid_actions_inp = featurize(
        embed_models,
        public_history=public_history,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        non_feature_dims=0,
    )

    assert hist_inp is not None
    assert isinstance(hist_inp, torch.Tensor)
    assert isinstance(private_inp, torch.Tensor)
    assert isinstance(valid_actions_inp, torch.Tensor)


def test_featurize_sequence(embed_models):
    """Test featurization with sequence inputs (non_feature_dims=1)."""
    private_inputs = [
        {
            "hand": [(1, 0), (2, 0)],
            "player_idx": 0,
            "trick": 1,
        },
        {
            "hand": [(3, 0), (4, 0)],
            "player_idx": 1,
            "trick": 1,
        },
    ]
    valid_actions = [[(1, 0)], [(3, 0)]]

    hist_inp, private_inp, valid_actions_inp = featurize(
        embed_models,
        public_history=None,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        non_feature_dims=1,
    )

    assert hist_inp is None
    assert isinstance(private_inp, torch.Tensor)
    assert isinstance(valid_actions_inp, torch.Tensor)
    assert len(private_inp.shape) == 2  # (sequence, features)


def test_featurize_batch_sequence(embed_models):
    """Test featurization with batch of sequences (non_feature_dims=2)."""
    private_inputs = [
        [
            {
                "hand": [(1, 0)],
                "player_idx": 0,
                "trick": 1,
            }
        ],
        [
            {
                "hand": [(2, 0)],
                "player_idx": 1,
                "trick": 1,
            }
        ],
    ]
    valid_actions = [[[(1, 0)]], [[(2, 0)]]]

    hist_inp, private_inp, valid_actions_inp = featurize(
        embed_models,
        public_history=None,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        non_feature_dims=2,
    )

    assert hist_inp is None
    assert isinstance(private_inp, torch.Tensor)
    assert isinstance(valid_actions_inp, torch.Tensor)
    assert len(private_inp.shape) == 3  # (batch, sequence, features)


def test_featurize_invalid_dims():
    """Test that invalid non_feature_dims raises assertion error."""
    with pytest.raises(AssertionError):
        featurize(
            {},
            public_history=None,
            private_inputs={},
            valid_actions=[],
            non_feature_dims=3,  # Invalid value
        )


def test_featurize_padding(embed_models):
    """Test that sequences of different lengths are properly padded."""
    private_inputs = [
        {
            "hand": [(1, 0), (2, 0)],  # 2 cards
            "player_idx": 0,
            "trick": 1,
        },
        {
            "hand": [(3, 0)],  # 1 card
            "player_idx": 1,
            "trick": 1,
        },
    ]
    valid_actions = [[(1, 0), (2, 0)], [(3, 0)]]

    hist_inp, private_inp, valid_actions_inp = featurize(
        embed_models,
        public_history=None,
        private_inputs=private_inputs,
        valid_actions=valid_actions,
        non_feature_dims=1,
    )

    # Check that padding was applied
    assert valid_actions_inp.shape[1] == 2  # Max length of valid actions
    assert torch.any(valid_actions_inp == -1)  # Padding value present
