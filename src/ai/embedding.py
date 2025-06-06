from functools import cached_property

import torch
from torch import nn

from ai.hyperparams import Hyperparams
from ai.utils import MLP
from game.settings import Settings


def maxpool(x, mask, *, dim, check_finite=True):
    x = x.masked_fill(
        mask,
        -float("inf"),
    )
    out = torch.max(x, dim=dim)[0]
    if check_finite:
        assert out.isfinite().all()

    return out


class PaddedEmbed(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        output_dim: int,
        dropout: float,
        onnx: bool,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed: nn.Module = nn.Embedding(
            num_embeddings + 1, output_dim, padding_idx=0
        )

        self.output_dim = output_dim
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.onnx = onnx

    def forward(self, x):
        if self.onnx:
            x = torch.clamp(x, -1, self.num_embeddings - 1)
        else:
            assert (x >= -1).all()
            assert (x < self.num_embeddings).all(), (
                x.max().item(),
                self.num_embeddings,
            )

        x = self.embed((x + 1).to(torch.int32))

        if self.dropout:
            x = self.dropout(x)

        return x


class CardModel(nn.Module):
    def __init__(
        self,
        settings: Settings,
        output_dim: int,
        dropout: float,
        onnx: bool,
    ):
        super().__init__()

        self.trump_suit = settings.num_side_suits
        self.trump_suit_delta = settings.side_suit_length + settings.use_nosignal

        num_ranks = settings.side_suit_length
        if settings.use_trump_suit:
            num_ranks += settings.trump_suit_length
        num_ranks += settings.use_nosignal  # to handle nosignal

        # +1 to handle nosignal
        num_suits = settings.num_suits + settings.use_nosignal

        self.onnx = onnx
        self.rank_embed = PaddedEmbed(
            num_ranks,
            output_dim,
            dropout,
            onnx,
        )
        self.suit_embed = PaddedEmbed(num_suits, output_dim, dropout, onnx)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.onnx:
            assert x.shape[-1] == 2
        rank = x[..., 0]
        suit = x[..., 1]

        if self.onnx:
            mask = (suit == self.trump_suit).to(rank.dtype)
            delta = mask * self.trump_suit_delta
            rank = rank + delta
        else:
            rank = torch.where(
                suit == self.trump_suit, rank + self.trump_suit_delta, rank
            )

        x = self.rank_embed(rank) + self.suit_embed(suit)
        return x


class TrickModel(nn.Module):
    def __init__(
        self, embed_dim: int, embed_dropout: float, onnx: bool, settings: Settings
    ):
        super().__init__()
        self.use_drafting = settings.use_drafting
        if self.use_drafting:
            self.draft_delta = settings.num_tricks
            self.draft_phase = settings.get_phase_idx("draft")
            num_embeddings = settings.num_tricks + settings.num_draft_tricks
        else:
            num_embeddings = settings.num_tricks

        self.embed = PaddedEmbed(num_embeddings, embed_dim, embed_dropout, onnx)
        self.onnx = onnx

    def forward(self, trick, phase):
        if self.use_drafting:
            if self.onnx:
                mask = (phase == self.draft_phase).to(trick.dtype)
                delta = mask * self.draft_delta
                trick = trick + delta
            else:
                trick = torch.where(
                    phase == self.draft_phase, trick + self.draft_delta, trick
                )

        return self.embed(trick)


class HandModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        card_model: CardModel,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
        onnx: bool,
    ):
        super().__init__()
        self.card_model = card_model
        self.output_dim = output_dim
        input_dim = self.card_model.output_dim
        self.layer_norm = nn.LayerNorm(input_dim)
        self.mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
        )
        self.onnx = onnx

    @cached_property
    def hand_params(self):
        return list(self.mlp.parameters())

    def forward(self, x, check_finite=True):
        mask = (x[..., 0] == -1).unsqueeze(-1)
        x = self.card_model(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = maxpool(x, mask, dim=-2, check_finite=check_finite and not self.onnx)
        return x


class TasksModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        task_model: PaddedEmbed,
        player_model: PaddedEmbed,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
        onnx: bool,
    ):
        super().__init__()
        self.task_model = task_model
        self.player_model = player_model
        self.output_dim = output_dim
        input_dim = self.task_model.output_dim + self.player_model.output_dim
        self.layer_norm = nn.LayerNorm(input_dim)
        self.mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
        )
        self.onnx = onnx

    @cached_property
    def tasks_params(self):
        return list(self.mlp.parameters())

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
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = maxpool(x, mask, dim=-2, check_finite=check_finite and not self.onnx)
        return x


class ActionModel(nn.Module):
    def __init__(
        self,
        card_model: CardModel,
        task_model: PaddedEmbed,
        onnx: bool,
        settings: Settings,
    ):
        super().__init__()
        self.card_model = card_model
        self.output_dim = card_model.output_dim
        self.use_drafting = settings.use_drafting
        if self.use_drafting:
            self.task_model = task_model
            assert card_model.output_dim == task_model.output_dim
        self.onnx = onnx

    def forward(self, x, single_step):
        if not self.use_drafting:
            return self.card_model(x)

        if self.onnx:
            assert single_step

        if single_step:
            # x = (N, 2) or (N, A, 2)
            # We need to support heterogeneous phases in the batch dim
            # for tree search.
            if self.onnx:
                is_draft = (x[:, 1:2] if len(x.shape) == 2 else x[:, :1, 1:2]) == -1
                y = torch.where(
                    is_draft,
                    self.task_model(x[..., 0]),
                    self.card_model(x),
                )
            else:
                is_draft_arr = (x[:, 1] if len(x.shape) == 2 else x[:, 0, 1]) == -1
                y = torch.empty(x.shape[:-1] + (self.output_dim,), device=x.device)
                for is_draft in is_draft_arr.unique().tolist():
                    mask = is_draft_arr == is_draft
                    if is_draft:
                        y[mask] = self.task_model(x[mask, ..., 0])
                    else:
                        y[mask] = self.card_model(x[mask])
            return y

        # x = (N, T, 2) or (N, T, A, 2)
        x_ex = x[0, :, 1] if len(x.shape) == 3 else x[0, :, 0, 1]
        draft_length = (x_ex >= 0).int().argmax().item()
        task_embed = self.task_model(x[:, :draft_length, ..., 0])
        card_embed = self.card_model(x[:, draft_length:])
        return torch.cat([task_embed, card_embed], dim=1)


def get_embed_models(
    hp: Hyperparams,
    settings: Settings,
    onnx: bool = False,
) -> dict[str, nn.Module]:
    # +1 b/c we need to encode the unassigned task.
    player_model = PaddedEmbed(
        settings.num_players + (1 if settings.use_drafting else 0),
        hp.embed_dim,
        hp.embed_dropout,
        onnx,
    )
    card_model = CardModel(
        settings,
        hp.embed_dim,
        hp.embed_dropout,
        onnx,
    )
    hand_model = HandModel(
        hp.hand_embed_dim,
        card_model,
        hidden_dim=hp.hand_hidden_dim,
        num_hidden_layers=hp.hand_num_hidden_layers,
        dropout=hp.hand_dropout,
        onnx=onnx,
    )
    # +1 b/c we need to encode the nodraft action.
    task_model = PaddedEmbed(
        settings.num_task_defs + (1 if settings.use_drafting else 0),
        hp.embed_dim,
        hp.embed_dropout,
        onnx,
    )
    tasks_model = TasksModel(
        hp.tasks_embed_dim,
        task_model,
        player_model,
        hp.tasks_hidden_dim,
        hp.tasks_num_hidden_layers,
        hp.tasks_dropout,
        onnx,
    )
    action_model: nn.Module = ActionModel(
        card_model,
        task_model,
        onnx,
        settings,
    )

    ret = {
        "player": player_model,
        "trick": TrickModel(
            hp.embed_dim,
            hp.embed_dropout,
            onnx,
            settings=settings,
        ),
        "turn": PaddedEmbed(
            settings.num_players,
            hp.embed_dim,
            hp.embed_dropout,
            onnx,
        ),
        "card": card_model,
        "action": action_model,
        "task": task_model,
        "tasks": tasks_model,
        "phase": PaddedEmbed(
            settings.num_phases,
            hp.embed_dim,
            hp.embed_dropout,
            onnx,
        ),
        "hand": hand_model,
    }

    return ret
