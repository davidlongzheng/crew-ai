from functools import cached_property
from typing import cast

import pandas as pd
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..game.settings import Settings
from .aux_info import get_aux_info_spec
from .embedding import (
    CardModel,
    HandModel,
    PaddedEmbed,
    TasksModel,
    get_embed_models,
)
from .hyperparams import Hyperparams
from .utils import MLP


class HistoryModel(nn.Module):
    def __init__(
        self,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        card_embed: nn.Module,
        turn_embed: nn.Module,
        phase_embed: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
        use_tasks: bool,
        tasks_embed_dim: int,
        use_tformer: bool,
    ):
        super().__init__()
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.card_embed = card_embed
        self.turn_embed = turn_embed
        self.phase_embed = phase_embed
        self.use_tasks = use_tasks

        input_dim = 5 * embed_dim
        if use_tasks:
            input_dim += tasks_embed_dim

        self.use_tformer = use_tformer
        if use_tformer:
            self.input_embed = nn.Linear(input_dim, hidden_dim)
            self.tformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    batch_first=True,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    norm_first=True,
                ),
                num_layers=num_layers,
            )
            self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None

            self.register_buffer("memory", torch.zeros(1, 1, hidden_dim))
        else:
            self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=(dropout if num_layers > 1 else 0.0),
            )
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.state: torch.Tensor | None = None
        self.single_step = False

    @cached_property
    def hist_params(self):
        return (
            list(self.tformer.parameters())
            if self.use_tformer
            else list(self.lstm.parameters())
        )

    def start_single_step(self):
        self.single_step = True
        self.state = None

    def stop_single_step(self):
        self.single_step = False
        self.state = None

    def forward(
        self,
        hist_inps: TensorDict,
        seq_lengths: Tensor | None = None,
        tasks_embed: Tensor | None = None,
    ) -> Tensor:
        assert (seq_lengths is not None) == (len(hist_inps["player_idxs"].shape) == 2)
        player_embed = self.player_embed(hist_inps["player_idxs"])
        trick_embed = self.trick_embed(hist_inps["tricks"])
        card_embed = self.card_embed(hist_inps["cards"])
        turn_embed = self.turn_embed(hist_inps["turns"])
        phase_embed = self.phase_embed(hist_inps["phases"])
        inps = [player_embed, trick_embed, card_embed, turn_embed, phase_embed]
        if self.use_tasks:
            assert tasks_embed is not None
            inps.append(tasks_embed)
        else:
            assert tasks_embed is None
        x = torch.cat(inps, dim=-1)
        if self.single_step:
            assert not self.training
            assert len(x.shape) <= 2

        if not self.single_step and self.state is not None:
            raise Exception(
                "Forgot to stop_single_step() before going back to multi-step mode."
            )

        if self.use_tformer:
            x = self.input_embed(x)

            if self.single_step:
                # F or B, F
                x = x.unsqueeze(dim=-2)
                if self.state is not None:
                    x = torch.cat([self.state, x], dim=-2)
                self.state = x

            assert len(x.shape) >= 2
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(-2), device=x.device
            )
            x = self.tformer(
                x,
                self.memory.expand(x.size(0), -1, -1),
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )

            if self.single_step:
                x = x[..., -1, :].squeeze(dim=-2)

            if self.layer_norm:
                x = self.layer_norm(x)
        else:
            if self.layer_norm:
                x = self.layer_norm(x)

            if seq_lengths is not None:
                x = pack_padded_sequence(
                    x, seq_lengths.to("cpu"), enforce_sorted=False, batch_first=True
                )  # type: ignore
            elif self.single_step:
                x = x.unsqueeze(dim=-2)

            x, state = self.lstm(x, self.state)

            if seq_lengths is not None:
                x, _ = pad_packed_sequence(x, padding_value=0.0, batch_first=True)  # type: ignore
            elif self.single_step:
                x = x.squeeze(dim=-2)

            # Only save state if we are using the model one action
            # at a time.
            if self.single_step:
                self.state = state

        x = self.fc(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class BackboneModel(nn.Module):
    def __init__(
        self,
        hist_model: HistoryModel,
        hand_embed: HandModel,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        turn_embed: nn.Module,
        phase_embed: nn.Module,
        tasks_embed: TasksModel,
        embed_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
        use_skip: bool,
        use_resid: bool,
        use_final_layer_norm: bool,
    ):
        super().__init__()
        self.hist_model = hist_model
        self.hand_embed = hand_embed
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.turn_embed = turn_embed
        self.phase_embed = phase_embed
        self.tasks_embed = tasks_embed
        self.use_skip = use_skip
        self.use_resid = use_resid
        self.output_dim = output_dim
        if self.use_skip:
            self.output_dim += self.hist_model.output_dim
        input_dim = (
            self.hist_model.output_dim
            + embed_dim * 4
            + self.tasks_embed.output_dim
            + self.hand_embed.output_dim
        )
        if self.use_resid:
            assert not self.use_skip
            self.input_embed = nn.Linear(input_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else None
            self.mlp = MLP(
                output_dim,
                hidden_dim,
                output_dim,
                num_hidden_layers,
                dropout=dropout,
            )
        else:
            self.input_embed = None
            self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
            self.mlp = MLP(
                input_dim,
                hidden_dim,
                output_dim,
                num_hidden_layers,
                dropout=dropout,
            )

        self.final_layer_norm = (
            nn.LayerNorm(output_dim) if use_final_layer_norm else None
        )

    @cached_property
    def mlp_params(self):
        ret = list(self.mlp.parameters())
        if self.input_embed:
            ret += list(self.input_embed.parameters())
        return ret

    def start_single_step(self):
        self.hist_model.start_single_step()

    def stop_single_step(self):
        self.hist_model.stop_single_step()

    def forward(
        self,
        hist_inps: TensorDict,
        private_inps: TensorDict,
        task_idxs: Tensor,
        seq_lengths: Tensor | None = None,
    ) -> Tensor:
        tasks_embed = self.tasks_embed(task_idxs)
        hist_embed: Tensor = self.hist_model(
            hist_inps=hist_inps,
            seq_lengths=seq_lengths,
            tasks_embed=(tasks_embed if self.hist_model.use_tasks else None),
        )
        hand_embed: Tensor = self.hand_embed(private_inps["hand"])
        player_embed = self.player_embed(private_inps["player_idx"])
        trick_embed = self.trick_embed(private_inps["trick"])
        turn_embed = self.turn_embed(private_inps["turn"])
        phase_embed = self.phase_embed(private_inps["phase"])
        private_embed = torch.cat(
            [player_embed, trick_embed, turn_embed, phase_embed], dim=-1
        )

        inps = [hist_embed, private_embed, tasks_embed, hand_embed]
        x = torch.cat(inps, dim=-1)

        if self.use_resid:
            x = self.input_embed(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        if self.use_resid:
            x = x + self.mlp(x)
        else:
            x = self.mlp(x)

        if self.final_layer_norm:
            x = self.final_layer_norm(x)

        if self.use_skip:
            x = torch.cat([x, hist_embed], dim=-1)

        return x


class PolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        card_embed: CardModel,
        phase_embed: PaddedEmbed,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
        use_layer_norm: bool,
        query_dim: int,
        num_tricks: int,
        signal_prior: str,
    ):
        super().__init__()
        self.card_embed = card_embed
        self.phase_embed = phase_embed
        self.query_model = nn.Linear(input_dim, query_dim)
        input_dim = card_embed.output_dim + phase_embed.output_dim
        self.key_layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
        self.key_model = MLP(
            input_dim,
            hidden_dim,
            query_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        self.query_dim = query_dim
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # pre-cache p_signal
        assert signal_prior in ["exp", "lin"]
        trick = torch.arange(num_tricks).float()
        num_tricks_left = num_tricks - trick
        if signal_prior == "exp":
            decay = 0.6
            self.register_buffer("p_signal", (1 - decay) / (1 - decay**num_tricks_left))
        else:
            self.register_buffer("p_signal", 2.0 / (num_tricks_left + 1.0))

    @cached_property
    def head_params(self):
        return list(self.key_model.parameters()) + list(self.query_model.parameters())

    def forward(
        self,
        backbone_embed: Tensor,
        valid_actions: Tensor,
        phase: Tensor,
        trick: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """valid_actions=(...,A,2)"""
        valid_actions_embed = self.card_embed(valid_actions)
        phase_embed = self.phase_embed(phase)
        phase_embed = phase_embed.unsqueeze(-2).expand(
            valid_actions_embed.shape[:-1] + (phase_embed.shape[-1],)
        )
        key_input = torch.cat([valid_actions_embed, phase_embed], dim=-1)
        if self.key_layer_norm:
            key_input = self.key_layer_norm(key_input)

        query = self.query_model(backbone_embed)
        key = self.key_model(key_input)
        # (..., A)
        attn_score = torch.einsum("...q,...vq->...v", query, key) / self.query_dim**0.5
        attn_score = attn_score.masked_fill(
            valid_actions[..., 0] == -1,
            -float("inf"),
        )

        # On phase == "signal", we want to upweight the "nosignal" choice
        # relative to the other choices based on what trick we're on and
        # how many other tricks there are.

        # (...)
        signal_phase = phase == 1
        # (...)
        n_valid_actions = (valid_actions[..., 0] != -1).sum(dim=-1).float()
        p_signal = self.p_signal[trick.long()]
        eps = torch.tensor(1e-5)
        no_signal_wgt = torch.log(
            torch.maximum((1 - p_signal) / p_signal * (n_valid_actions - 1), eps)
        )
        no_signal_wgt = torch.where(n_valid_actions == 1, 0.0, no_signal_wgt)

        # The nosignal is at the 0th index.
        attn_score[..., 0] += torch.where(signal_phase, no_signal_wgt, 0.0)

        probs = self.softmax(attn_score)
        log_probs = self.log_softmax(attn_score)

        return probs, log_probs


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        # Value Head
        self.fc = nn.Linear(input_dim, 1)

    def forward(
        self,
        backbone_embed: Tensor,
    ) -> Tensor:
        value = self.fc(backbone_embed).squeeze(-1)
        return value


class AuxInfoHead(nn.Module):
    def __init__(self, input_dim: int, spec: pd.DataFrame):
        super().__init__()
        out_dim = int(spec["num_cat"].fillna(1.0).sum())
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(
        self,
        backbone_embed: Tensor,
    ) -> tuple[Tensor]:
        return self.fc(backbone_embed)


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        backbone_model: BackboneModel,
        policy_head: PolicyHead,
        value_head: ValueHead,
        aux_info_head: AuxInfoHead | None,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.policy_head = policy_head
        self.value_head = value_head
        self.aux_info_head = aux_info_head

    @cached_property
    def param_groups(self):
        return {
            "policy_head": self.policy_head.head_params,
            "value_head": list(self.value_head.parameters()),
            "hist": self.backbone_model.hist_model.hist_params,
            "mlp": self.backbone_model.mlp_params,
            # phase, trick, turn
            "basic_embed": list(self.backbone_model.phase_embed.parameters())
            + list(self.backbone_model.player_embed.parameters())
            + list(self.backbone_model.trick_embed.parameters())
            + list(self.backbone_model.turn_embed.parameters()),
            "hand": list(self.backbone_model.hand_embed.hand_params),
            "card": list(self.backbone_model.hand_embed.card_model.parameters()),
            "task": list(self.backbone_model.tasks_embed.task_model.parameters()),
            "tasks": list(self.backbone_model.tasks_embed.tasks_params),
        }

    def start_single_step(self):
        self.backbone_model.start_single_step()

    def stop_single_step(self):
        self.backbone_model.stop_single_step()

    def forward(self, inps):
        backbone_embed = self.backbone_model(
            inps["hist"], inps["private"], inps["task_idxs"], inps["seq_lengths"]
        )
        aux_info_pred = (
            self.aux_info_head(backbone_embed) if self.aux_info_head else None
        )
        policy_pred = self.policy_head(
            backbone_embed,
            inps["valid_actions"],
            inps["private"]["phase"],
            inps["private"]["trick"],
        )
        value_pred = self.value_head(backbone_embed)

        return policy_pred, value_pred, aux_info_pred


def get_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    ret: dict[str, nn.Module] = {}
    embed_models = get_embed_models(hp, settings)
    hist_model = HistoryModel(
        embed_models["player"],
        embed_models["trick"],
        embed_models["card"],
        embed_models["turn"],
        embed_models["phase"],
        hp.embed_dim,
        hp.hist_hidden_dim,
        hp.hist_num_layers,
        hp.hist_output_dim,
        hp.hist_dropout,
        hp.hist_use_layer_norm,
        hp.hist_use_tasks,
        hp.tasks_embed_dim,
        hp.hist_use_tformer,
    )
    backbone_model = BackboneModel(
        hist_model,
        cast(HandModel, embed_models["hand"]),
        embed_models["player"],
        embed_models["trick"],
        embed_models["turn"],
        embed_models["phase"],
        cast(TasksModel, embed_models["tasks"]),
        hp.embed_dim,
        hp.backbone_hidden_dim,
        hp.backbone_num_hidden_layers,
        hp.backbone_output_dim,
        hp.backbone_dropout,
        hp.backbone_use_layer_norm,
        hp.backbone_use_skip,
        hp.backbone_use_resid,
        hp.backbone_use_final_layer_norm,
    )
    if hp.aux_info_coef:
        aux_info_head = AuxInfoHead(
            input_dim=backbone_model.output_dim,
            spec=get_aux_info_spec(settings),
        )
    else:
        aux_info_head = None

    policy_head = PolicyHead(
        backbone_model.output_dim,
        cast(CardModel, embed_models["card"]),
        cast(PaddedEmbed, embed_models["phase"]),
        hp.policy_hidden_dim,
        hp.policy_num_hidden_layers,
        hp.policy_dropout,
        hp.policy_use_layer_norm,
        hp.policy_query_dim,
        settings.num_tricks,
        hp.policy_signal_prior,
    )
    value_head = ValueHead(
        input_dim=backbone_model.output_dim,
    )
    ret["pv"] = PolicyValueModel(
        backbone_model,
        policy_head,
        value_head,
        aux_info_head,
    )
    return ret
