import torch
from torch import nn

from ..game.settings import Settings
from .hyperparams import Hyperparams


class CardModel(nn.Module):
    def __init__(
        self,
        num_ranks: int,
        num_suits: int,
        embed_dim: int,
    ):
        super().__init__()
        self.rank_embed = nn.Embedding(num_ranks + 1, embed_dim, padding_idx=num_ranks)
        self.suit_embed = nn.Embedding(num_suits + 1, embed_dim, padding_idx=num_suits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2
        x = self.rank_embed(x[..., 0]) + self.suit_embed(x[..., 1])
        return x


class HandModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        card_model: CardModel,
        hidden_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.card_model = card_model

        assert num_hidden_layers >= 1
        fcs = [
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            fcs += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        fcs.append(nn.Linear(hidden_dim, embed_dim))
        self.fc = nn.Sequential(*fcs)

    def forward(self, x):
        x = self.card_model(x)
        x = self.fc(x)
        return torch.max(x, dim=-2)[0]


def get_embed_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    card_model = CardModel(settings.num_ranks, settings.num_suits, hp.embed_dim)
    hand_model = HandModel(
        hp.embed_dim,
        card_model,
        hidden_dim=hp.hand_hidden_dim,
        num_hidden_layers=hp.hand_num_hidden_layers,
    )
    return {
        "player": nn.Embedding(
            settings.num_players + 1,
            hp.embed_dim,
            padding_idx=settings.num_players,
        ),
        "trick": nn.Embedding(
            settings.num_tricks + 1,
            hp.embed_dim,
            padding_idx=settings.num_tricks,
        ),
        "card": card_model,
        "hand": hand_model,
    }
