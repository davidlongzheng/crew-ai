import pytest
import torch

from ...game.settings import Settings
from ..embedding import CardModel, HandModel, PaddedEmbed, get_embed_models
from ..hyperparams import Hyperparams


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def hp() -> Hyperparams:
    return Hyperparams(embed_dim=16, hand_hidden_dim=32, hand_num_hidden_layers=2)


@pytest.fixture
def embed_models(hp, settings):
    return get_embed_models(hp, settings)


def get_rand_cards(shape, settings: Settings) -> torch.Tensor:
    cards = torch.stack(
        [
            torch.randint(0, settings.num_ranks, shape),
            torch.randint(0, settings.num_suits, shape),
        ],
        dim=-1,
    )
    return cards


def test_card_model_initialization(settings: Settings, embed_models, hp):
    model = embed_models["card"]

    assert isinstance(model.rank_embed, PaddedEmbed)
    assert isinstance(model.suit_embed, PaddedEmbed)
    assert model.rank_embed.embed.num_embeddings == settings.num_ranks + 1
    assert model.suit_embed.embed.num_embeddings == settings.num_suits + 1
    assert model.rank_embed.embed.embedding_dim == hp.embed_dim
    assert model.suit_embed.embed.embedding_dim == hp.embed_dim


def test_card_model_forward(embed_models, settings, hp):
    model = embed_models["card"]
    model.eval()

    # Test single card
    single_card = torch.tensor([[3, 2]])  # rank 4, suit 3
    output = model(single_card)
    assert output.shape == (1, hp.embed_dim)

    # Test batch of cards
    batch_size = 32
    cards = get_rand_cards((batch_size, 1), settings)
    output = model(cards)
    assert output.shape == (batch_size, 1, hp.embed_dim)

    # Test padding
    padded_card = torch.tensor([[-1, -1]])
    output = model(padded_card)
    assert output.shape == (1, hp.embed_dim)
    assert torch.all(output == 0)  # Padding should give zero embeddings


def test_hand_model_initialization(settings: Settings, hp, embed_models):
    model = embed_models["hand"]

    assert isinstance(model.card_model, CardModel)
    assert isinstance(model.mlp, torch.nn.Sequential)
    # Check number of layers (input layer + hidden layers + output layer)
    assert (
        len([m for m in model.mlp if isinstance(m, torch.nn.Linear)])
        == hp.hand_num_hidden_layers + 1
    )


def test_hand_model_forward(settings: Settings, hp, embed_models):
    model = embed_models["hand"]
    model.eval()

    # Test single hand with multiple cards
    num_cards = 5
    hand = get_rand_cards((1, num_cards), settings)
    output = model(hand)
    assert output.shape == (1, hp.embed_dim)

    # Test batch of hands
    batch_size = 32
    hands = get_rand_cards((batch_size, num_cards), settings)
    output = model(hands)
    assert output.shape == (batch_size, hp.embed_dim)

    # Test padding
    padded_hand = torch.full((1, num_cards, 2), -1)
    padded_hand[0, 0] = torch.tensor([0, 0])
    output1 = model(padded_hand)
    output2 = model(padded_hand[:, :1])
    assert output1.shape == (1, hp.embed_dim)
    assert torch.allclose(output1, output2, atol=1e-4)


def test_get_embed_models(settings: Settings, hp: Hyperparams):
    models = get_embed_models(hp, settings)

    assert len(models) == 5
    assert all(key in models for key in ["player", "trick", "turn", "card", "hand"])

    # Test player embedding
    assert isinstance(models["player"], PaddedEmbed)
    assert models["player"].embed.num_embeddings == settings.num_players + 1
    assert models["player"].embed.embedding_dim == hp.embed_dim

    # Test trick embedding
    assert isinstance(models["trick"], PaddedEmbed)
    assert models["trick"].embed.num_embeddings == settings.num_tricks + 1
    assert models["trick"].embed.embedding_dim == hp.embed_dim

    # Test turn embedding
    assert isinstance(models["turn"], PaddedEmbed)
    assert models["turn"].embed.num_embeddings == settings.num_players + 1
    assert models["turn"].embed.embedding_dim == hp.embed_dim

    # Test card model
    assert isinstance(models["card"], CardModel)

    # Test hand model
    assert isinstance(models["hand"], HandModel)
