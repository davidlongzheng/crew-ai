import torch
from torch import nn

from ..game.settings import Settings
from .hyperparams import Hyperparams
from .utils import make_mlp


class PaddedEmbed(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings + 1, embed_dim, padding_idx=0)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed((x + 1).to(torch.int32))
        if self.dropout:
            x = self.dropout(x)

        return x


class CardModel(nn.Module):
    def __init__(
        self,
        num_ranks: int,
        num_suits: int,
        embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.rank_embed = PaddedEmbed(num_ranks, embed_dim, dropout)
        self.suit_embed = PaddedEmbed(num_suits, embed_dim, dropout)

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
        use_layer_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.card_model = card_model
        self.mlp = make_mlp(
            embed_dim,
            hidden_dim,
            embed_dim,
            num_hidden_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x):
        mask = x[..., 0] == -1
        x = self.card_model(x)
        x = self.mlp(x)
        x = x.masked_fill(
            mask.unsqueeze(-1),
            -float("inf"),
        )
        x = torch.max(x, dim=-2)[0]
        assert not torch.isneginf(x).any()
        return x


def get_embed_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    card_model = CardModel(
        settings.num_ranks, settings.num_suits, hp.embed_dim, hp.embed_dropout
    )
    hand_model = HandModel(
        hp.embed_dim,
        card_model,
        hidden_dim=hp.hand_hidden_dim,
        num_hidden_layers=hp.hand_num_hidden_layers,
        use_layer_norm=hp.hand_use_layer_norm,
        dropout=hp.hand_dropout,
    )
    return {
        "player": PaddedEmbed(
            settings.num_players,
            hp.embed_dim,
            hp.embed_dropout,
        ),
        "trick": PaddedEmbed(
            settings.num_tricks,
            hp.embed_dim,
            hp.embed_dropout,
        ),
        "turn": PaddedEmbed(
            settings.num_players,
            hp.embed_dim,
            hp.embed_dropout,
        ),
        "card": card_model,
        "hand": hand_model,
    }
