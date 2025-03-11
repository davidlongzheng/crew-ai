import pytest
import torch

from ...game.settings import Settings
from ..embedding import CardModel, HandModel, get_embed_models
from ..hyperparams import Hyperparams


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def hyperparams() -> Hyperparams:
    return Hyperparams(embed_dim=16, hand_hidden_dim=32, hand_num_hidden_layers=2)


def get_rand_cards(shape, settings: Settings) -> torch.Tensor:
    cards = torch.stack(
        [
            torch.randint(0, settings.num_ranks, shape),
            torch.randint(0, settings.num_suits, shape),
        ],
        dim=-1,
    )
    return cards


def test_card_model_initialization(settings: Settings):
    embed_dim = 16
    model = CardModel(settings.num_ranks, settings.num_suits, embed_dim)

    assert isinstance(model.rank_embed, torch.nn.Embedding)
    assert isinstance(model.suit_embed, torch.nn.Embedding)
    assert model.rank_embed.num_embeddings == settings.num_ranks
    assert model.suit_embed.num_embeddings == settings.num_suits
    assert model.rank_embed.embedding_dim == embed_dim
    assert model.suit_embed.embedding_dim == embed_dim


def test_card_model_forward(settings: Settings):
    embed_dim = 16
    model = CardModel(settings.num_ranks, settings.num_suits, embed_dim)

    # Test single card
    single_card = torch.tensor([[3, 2]])  # rank 4, suit 3
    output = model(single_card)
    assert output.shape == (1, embed_dim)

    # Test batch of cards
    batch_size = 32
    cards = get_rand_cards((batch_size, 1), settings)
    output = model(cards)
    assert output.shape == (batch_size, 1, embed_dim)


def test_hand_model_initialization(settings: Settings):
    embed_dim = 16
    hidden_dim = 32
    num_hidden_layers = 2
    card_model = CardModel(settings.num_ranks, settings.num_suits, embed_dim)

    model = HandModel(
        embed_dim=embed_dim,
        card_model=card_model,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )

    assert isinstance(model.card_model, CardModel)
    assert isinstance(model.fc, torch.nn.Sequential)
    # Check number of layers (input layer + hidden layers + output layer)
    assert (
        len([m for m in model.fc if isinstance(m, torch.nn.Linear)])
        == num_hidden_layers + 1
    )


def test_hand_model_forward(settings: Settings):
    embed_dim = 16
    hidden_dim = 32
    num_hidden_layers = 2
    card_model = CardModel(settings.num_ranks, settings.num_suits, embed_dim)

    model = HandModel(
        embed_dim=embed_dim,
        card_model=card_model,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )

    # Test single hand with multiple cards
    num_cards = 5
    hand = get_rand_cards((1, num_cards), settings)
    output = model(hand)
    assert output.shape == (1, embed_dim)  # max value across cards

    # Test batch of hands
    batch_size = 32
    hands = get_rand_cards((batch_size, num_cards), settings)
    output = model(hands)
    assert output.shape == (batch_size, embed_dim)


def test_get_embed_models(settings: Settings, hyperparams: Hyperparams):
    models = get_embed_models(hyperparams, settings)

    assert len(models) == 4
    assert all(key in models for key in ["player", "trick", "card", "hand"])

    # Test player embedding
    assert isinstance(models["player"], torch.nn.Embedding)
    assert models["player"].num_embeddings == settings.num_players
    assert models["player"].embedding_dim == hyperparams.embed_dim

    # Test trick embedding
    assert isinstance(models["trick"], torch.nn.Embedding)
    assert models["trick"].num_embeddings == settings.num_tricks
    assert models["trick"].embedding_dim == hyperparams.embed_dim

    # Test card model
    assert isinstance(models["card"], CardModel)

    # Test hand model
    assert isinstance(models["hand"], HandModel)
