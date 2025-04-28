import torch
import torch.nn.functional as F
from torch import nn

from ..game.settings import Settings
from ..game.tasks import get_task_defs
from .hyperparams import Hyperparams
from .utils import MLP

AGG_METHODS = ["maxpool", "sumpool", "avgpool"]


def thermometer_encode(x, num_classes):
    levels = torch.arange(num_classes, device=x.device)
    return (x.unsqueeze(-1) >= levels).to(torch.float32)


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
        embed_type: str,
    ):
        super().__init__()
        self.embed_type = embed_type
        self.num_embeddings = num_embeddings
        if self.embed_type == "embed":
            self.embed: nn.Module = nn.Embedding(
                num_embeddings + 1, output_dim, padding_idx=0
            )
        elif self.embed_type in ["one_hot", "thermo"]:
            self.embed = nn.Linear(num_embeddings + 1, output_dim)
        else:
            raise ValueError(self.embed_type)

        self.output_dim = output_dim
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.embed_type == "embed":
            x = self.embed((x + 1).to(torch.int32))
        elif self.embed_type == "one_hot":
            x = self.embed(
                F.one_hot((x + 1).long(), num_classes=self.num_embeddings + 1).float()
            )
        elif self.embed_type == "thermo":
            x = self.embed(
                thermometer_encode(
                    (x + 1).long(), num_classes=self.num_embeddings + 1
                ).float()
            )
        else:
            raise ValueError(self.embed_type)

        if self.dropout:
            x = self.dropout(x)

        return x


class CardModel(nn.Module):
    def __init__(
        self,
        max_suit_length: int,
        num_suits: int,
        output_dim: int,
        dropout: float,
        use_thermo: bool,
    ):
        super().__init__()
        self.max_suit_length = max_suit_length
        self.num_suits = num_suits
        self.rank_embed = PaddedEmbed(
            max_suit_length,
            output_dim,
            dropout,
            embed_type="thermo" if use_thermo else "embed",
        )
        self.suit_embed = PaddedEmbed(
            num_suits, output_dim, dropout, embed_type="embed"
        )
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


class TasksModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        task_model: PaddedEmbed,
        player_model: PaddedEmbed,
        hidden_dim: int,
        num_hidden_layers: int,
        use_layer_norm: bool,
        dropout: float,
        agg_method: str,
    ):
        super().__init__()
        self.task_model = task_model
        self.player_model = player_model
        self.output_dim = output_dim
        input_dim = self.task_model.output_dim + self.player_model.output_dim
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
        """Input: (..., K, 2)"""
        # (..., K, 1)
        mask = (x[..., 0] == -1).unsqueeze(-1)
        # (..., K, F)
        x = torch.cat(
            [
                self.task_model(x[..., 0]),
                self.player_model(x[..., 1]),
            ],
            dim=-1,
        )
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
        agg_method: str,
    ):
        super().__init__()
        self.hand_model = hand_model
        self.player_model = player_model
        self.input_dim = hand_model.output_dim + player_model.output_dim
        self.layer_norm = nn.LayerNorm(self.input_dim) if use_layer_norm else None
        self.output_dim = output_dim
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
        x = torch.cat(
            [x, player_embed.expand(*x.shape[:2], *player_embed.shape)], dim=-1
        )

        if self.layer_norm:
            x = self.layer_norm(x)

        x = self.mlp(x)
        x = aggregate(x, mask, method=self.agg_method, dim=-2)
        return x


def get_embed_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    player_model = PaddedEmbed(
        settings.num_players,
        hp.embed_dim,
        hp.embed_dropout,
        embed_type=("thermo" if hp.embed_use_thermo else "embed"),
    )
    # +1 to handle nosignal case.
    card_model = CardModel(
        settings.max_suit_length + 1,
        settings.num_suits + 1,
        hp.embed_dim,
        hp.embed_dropout,
        hp.embed_use_thermo,
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
    task_model = PaddedEmbed(
        len(get_task_defs(settings.bank)),
        hp.embed_dim,
        hp.embed_dropout,
        embed_type="embed",
    )
    tasks_model = TasksModel(
        hp.tasks_embed_dim,
        task_model,
        player_model,
        hp.tasks_hidden_dim,
        hp.tasks_num_hidden_layers,
        hp.tasks_use_layer_norm,
        hp.tasks_dropout,
        hp.tasks_agg_method,
    )
    ret = {
        "player": player_model,
        "trick": PaddedEmbed(
            settings.num_tricks,
            hp.embed_dim,
            hp.embed_dropout,
            embed_type=("thermo" if hp.embed_use_thermo else "embed"),
        ),
        "turn": PaddedEmbed(
            settings.num_players,
            hp.embed_dim,
            hp.embed_dropout,
            embed_type=("thermo" if hp.embed_use_thermo else "embed"),
        ),
        "card": card_model,
        "tasks": tasks_model,
        "phase": PaddedEmbed(
            settings.num_phases,
            hp.embed_dim,
            hp.embed_dropout,
            embed_type="embed",
        ),
    }

    ret["hand"] = hand_model

    return ret
