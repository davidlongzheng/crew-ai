import torch
from torch import nn

from ..game.settings import Settings
from .hyperparams import Hyperparams
from .utils import MLP

AGG_METHODS = ["maxpool", "sumpool", "avgpool"]


def aggregate(x, mask, *, method, dim, check_finite=True):
    assert method in AGG_METHODS, method

    x = x.masked_fill(
        mask,
        -float("inf") if method == "maxpool" else float("nan"),
    )

    if method == "maxpool":
        out = torch.max(x, dim=dim)[0]
    elif method == "sumpool":
        out = torch.nansum(x, dim=dim)
    elif method == "avgpool":
        num_notna = (~x.isnan()).sum(dim=dim)
        out = torch.where(
            num_notna > 0,
            torch.nansum(x, dim=dim) / num_notna.clamp(min=1),
            float("nan"),
        )
    else:
        raise ValueError(method)

    if check_finite:
        assert out.isfinite().all()

    return out


class PaddedEmbed(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings + 1, output_dim, padding_idx=0)
        self.dropout = None
        self.output_dim = output_dim
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
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.rank_embed = PaddedEmbed(num_ranks, output_dim, dropout)
        self.suit_embed = PaddedEmbed(num_suits, output_dim, dropout)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2
        x = self.rank_embed(x[..., 0]) + self.suit_embed(x[..., 1])
        return x


class HandModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        card_model: CardModel,
        hidden_dim: int,
        num_hidden_layers: int,
        use_layer_norm: bool,
        dropout: float,
        agg_method: str,
    ):
        super().__init__()
        self.card_model = card_model
        self.output_dim = output_dim
        input_dim = self.card_model.output_dim
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
        self.mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
        )
        assert agg_method in AGG_METHODS, agg_method
        self.agg_method = agg_method

    def forward(self, x, check_finite=True):
        mask = (x[..., 0] == -1).unsqueeze(-1)
        x = self.card_model(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.mlp(x)
        x = aggregate(
            x, mask, method=self.agg_method, dim=-2, check_finite=check_finite
        )
        return x


class HandsModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hand_model: HandModel,
        player_model: PaddedEmbed,
        hidden_dim: int,
        num_hidden_layers: int,
        use_layer_norm: bool,
        dropout: float,
        concat_inputs: bool,
        agg_method: str,
    ):
        super().__init__()
        self.hand_model = hand_model
        self.player_model = player_model
        self.input_dim = (
            hand_model.output_dim + player_model.output_dim
            if concat_inputs
            else max(hand_model.output_dim, player_model.output_dim)
        )
        self.layer_norm = nn.LayerNorm(self.input_dim) if use_layer_norm else None
        self.output_dim = output_dim
        self.concat_inputs = concat_inputs
        self.mlp = MLP(
            self.input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
        )
        assert agg_method in AGG_METHODS, agg_method
        self.agg_method = agg_method

    def forward(self, x):
        # inp = (N,T,P,H,2) out = (N,T,P,F)
        x = self.hand_model(x, check_finite=False)
        # For any players with no cards left, it's possible
        # for this to be non-finite
        mask = (~x[..., 0].isfinite()).unsqueeze(-1)
        x = x.masked_fill(
            mask,
            0.0,
        )

        # (P,)
        player_idx = torch.arange(x.shape[-2], dtype=torch.int8)
        # (P, F)
        player_embed = self.player_model(player_idx)
        if self.concat_inputs:
            x = torch.cat(
                [x, player_embed.expand(*x.shape[:2], *player_embed.shape)], dim=-1
            )
        else:
            x = nn.functional.pad(x, (0, self.input_dim - x.shape[-1]))
            player_embed = nn.functional.pad(
                player_embed, (0, self.input_dim - player_embed.shape[-1])
            )
            x = x + player_embed

        if self.layer_norm:
            x = self.layer_norm(x)

        x = self.mlp(x)
        x = aggregate(x, mask, method=self.agg_method, dim=-2)
        return x


def get_embed_models(
    hp: Hyperparams,
    settings: Settings,
    network_type: str,
) -> dict[str, nn.Module]:
    player_model = PaddedEmbed(
        settings.num_players,
        hp.embed_dim,
        hp.embed_dropout,
    )
    card_model = CardModel(
        settings.num_ranks, settings.num_suits, hp.embed_dim, hp.embed_dropout
    )
    hand_model = HandModel(
        hp.hand_embed_dim,
        card_model,
        hidden_dim=hp.hand_hidden_dim,
        num_hidden_layers=hp.hand_num_hidden_layers,
        use_layer_norm=hp.hand_use_layer_norm,
        dropout=hp.hand_dropout,
        agg_method=hp.hand_agg_method,
    )
    ret = {
        "player": player_model,
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
    }

    if network_type == "value":
        ret["hands"] = HandsModel(
            hp.hands_embed_dim,
            hand_model,
            player_model,
            hidden_dim=hp.hands_hidden_dim,
            num_hidden_layers=hp.hands_num_hidden_layers,
            use_layer_norm=hp.hands_use_layer_norm,
            dropout=hp.hands_dropout,
            concat_inputs=hp.hands_concat_inputs,
            agg_method=hp.hands_agg_method,
        )
    else:
        ret["hand"] = hand_model

    return ret
