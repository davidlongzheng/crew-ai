from functools import cached_property
from pathlib import Path
from typing import Callable, cast

import torch
from tensordict import TensorDict
from torch import Tensor, nn

from ai.embedding import (
    ActionModel,
    HandModel,
    PaddedEmbed,
    TasksModel,
    get_embed_models,
)
from ai.hyperparams import Hyperparams
from ai.utils import MLP
from game.settings import Settings
from game.utils import get_splits_and_phases


class BranchedLSTM(nn.Module):
    def __init__(self, make_lstm: Callable[[], nn.Module], settings: Settings):
        super().__init__()
        self.lstms = nn.ModuleList([make_lstm() for _ in range(settings.num_phases)])
        self.set_splits(settings)

    def set_splits(self, settings):
        self.splits, self.phases = get_splits_and_phases(settings)
        # Since this is processing historical values, everything is offset
        # by 1. We arbitrarily assign the first null value to the first split.
        self.splits[0] += 1
        self.splits[-1] -= 1

    def forward(self, x, state, phases, single_step):
        if single_step:
            phase = phases[0].item() if state is not None else self.phases[0]
            lstm = self.lstms[phase]
            return lstm(x, state)

        x_splits = torch.split(x, self.splits, dim=1)
        y_splits = []
        for phase, x_split in zip(self.phases, x_splits):
            y_split, state = self.lstms[phase](x_split, state)
            y_splits.append(y_split)
        y = torch.cat(y_splits, dim=1)

        return y, state


class BranchedFF(nn.Module):
    def __init__(self, make_ff: Callable[[], nn.Module], settings: Settings):
        super().__init__()
        self.ffs = nn.ModuleList([make_ff() for _ in range(settings.num_phases)])
        self.set_splits(settings)

    def set_splits(self, settings):
        splits, phases = get_splits_and_phases(settings)

        self.register_buffer(
            "phase_mask",
            torch.zeros(
                (settings.num_phases, settings.get_seq_length()), dtype=torch.bool
            ),
        )

        t = 0
        for split, phase in zip(splits, phases):
            self.phase_mask[phase, t : t + split] = True
            t += split
        assert t == settings.get_seq_length()

    def forward(self, x, phases):
        # phases = (N,) or (N, T)

        if len(phases.shape) == 1:
            phase = phases[0].item()
            ff = self.ffs[phase]
            return ff(x)

        y_splits = []
        for phase in range(len(self.ffs)):
            x_split = x[:, self.phase_mask[phase]]
            y_splits.append(self.ffs[phase](x_split))

        y_shape = x.shape[:-1] + (y_splits[0].size(-1),)
        y = torch.empty(y_shape, device=x.device)
        for phase in range(len(self.ffs)):
            y[:, self.phase_mask[phase]] = y_splits[phase]

        return y


class HistoryModel(nn.Module):
    def __init__(
        self,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        action_embed: nn.Module,
        turn_embed: nn.Module,
        phase_embed: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        tasks_embed_dim: int,
        phase_branch: bool,
        settings: Settings,
    ):
        super().__init__()
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.action_embed = action_embed
        self.turn_embed = turn_embed
        self.phase_embed = None if phase_branch else phase_embed
        self.phase_branch = phase_branch
        input_dim = (4 if phase_branch else 5) * embed_dim + tasks_embed_dim

        self.layer_norm = nn.LayerNorm(input_dim)
        make_lstm = lambda: nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        if phase_branch:
            self.lstm: nn.Module = BranchedLSTM(make_lstm, settings)
        else:
            self.lstm = make_lstm()
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.state: torch.Tensor | None = None
        self.single_step = False

    @cached_property
    def hist_params(self):
        return list(self.lstm.parameters())

    def start_single_step(self):
        self.single_step = True
        self.state = None

    def stop_single_step(self):
        self.single_step = False
        self.state = None

    def forward(
        self,
        hist_inps: TensorDict,
        tasks_embed: Tensor,
    ) -> Tensor:
        player_embed = self.player_embed(hist_inps["player_idx"])
        trick_embed = self.trick_embed(hist_inps["trick"], hist_inps["phase"])
        action_embed = self.action_embed(hist_inps["action"], self.single_step)
        turn_embed = self.turn_embed(hist_inps["turn"])
        inps = [
            player_embed,
            trick_embed,
            action_embed,
            turn_embed,
            tasks_embed,
        ]
        if not self.phase_branch:
            phase_embed = cast(nn.Module, self.phase_embed)(hist_inps["phase"])
            inps.append(phase_embed)
        x = torch.cat(inps, dim=-1)
        if self.single_step:
            assert not self.training
            assert len(x.shape) <= 2

        if not self.single_step and self.state is not None:
            raise Exception(
                "Forgot to stop_single_step() before going back to multi-step mode."
            )

        x = self.layer_norm(x)

        if self.single_step:
            x = x.unsqueeze(dim=-2)

        if self.phase_branch:
            x, state = self.lstm(x, self.state, hist_inps["phase"], self.single_step)
        else:
            x, state = self.lstm(x, self.state)

        if self.single_step:
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
        phase_branch: bool,
        settings: Settings,
    ):
        super().__init__()
        self.hist_model = hist_model
        self.hand_embed = hand_embed
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.turn_embed = turn_embed
        self.phase_branch = phase_branch
        self.phase_embed = None if phase_branch else phase_embed
        self.tasks_embed = tasks_embed
        self.output_dim = output_dim + self.hist_model.output_dim
        input_dim = (
            self.hist_model.output_dim
            + embed_dim * (3 if phase_branch else 4)
            + self.tasks_embed.output_dim
            + self.hand_embed.output_dim
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        make_mlp = lambda: MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        if phase_branch:
            self.mlp: nn.Module = BranchedFF(make_mlp, settings)
        else:
            self.mlp = make_mlp()

        self.final_layer_norm = nn.LayerNorm(output_dim)

    @cached_property
    def mlp_params(self):
        return list(self.mlp.parameters())

    def forward(
        self,
        hist_inps: TensorDict,
        private_inps: TensorDict,
    ) -> Tensor:
        tasks_embed = self.tasks_embed(private_inps["task_idxs"])
        hist_embed: Tensor = self.hist_model(hist_inps, tasks_embed)
        hand_embed: Tensor = self.hand_embed(private_inps["hand"])
        player_embed = self.player_embed(private_inps["player_idx"])
        trick_embed = self.trick_embed(private_inps["trick"], private_inps["phase"])
        turn_embed = self.turn_embed(private_inps["turn"])
        inps = [
            hist_embed,
            player_embed,
            trick_embed,
            turn_embed,
            tasks_embed,
            hand_embed,
        ]
        if not self.phase_branch:
            phase_embed = cast(nn.Module, self.phase_embed)(private_inps["phase"])
            inps.append(phase_embed)

        x = torch.cat(inps, dim=-1)
        x = self.layer_norm(x)
        if self.phase_branch:
            x = self.mlp(x, private_inps["phase"])
        else:
            x = self.mlp(x)
        if self.final_layer_norm:
            x = self.final_layer_norm(x)
        x = torch.cat([x, hist_embed], dim=-1)

        return x


class PolicyHead(nn.Module):
    def __init__(
        self,
        query_input_dim: int,
        action_embed: ActionModel,
        phase_embed: PaddedEmbed,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
        query_dim: int,
        signal_prior: str,
        phase_branch: bool,
        settings: Settings,
    ):
        super().__init__()

        self.action_embed = action_embed
        self.phase_embed = None if phase_branch else phase_embed
        self.phase_branch = phase_branch
        key_input_dim = action_embed.output_dim + (
            0 if phase_branch else phase_embed.output_dim
        )
        self.key_layer_norm = nn.LayerNorm(key_input_dim)
        make_key_model = lambda: MLP(
            key_input_dim,
            hidden_dim,
            query_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        if phase_branch:
            self.key_model: nn.Module = BranchedFF(make_key_model, settings)
        else:
            self.key_model = make_key_model()
        self.query_dim = query_dim
        make_query_model = lambda: nn.Linear(query_input_dim, query_dim)
        if phase_branch:
            self.query_model: nn.Module = BranchedFF(make_query_model, settings)
        else:
            self.query_model = make_query_model()
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.use_nosignal = settings.use_nosignal
        if settings.use_nosignal:
            # pre-cache p_signal
            self.signal_phase = settings.get_phase_idx("signal")
            assert signal_prior in ["exp", "lin"]
            trick = torch.arange(settings.num_tricks).float()
            num_tricks_left = settings.num_tricks - trick
            if signal_prior == "exp":
                decay = 0.6
                self.register_buffer(
                    "p_signal", (1 - decay) / (1 - decay**num_tricks_left)
                )
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
        """valid_actions=(...,A,3)"""

        single_step = len(valid_actions.shape) == 3
        valid_actions_embed = self.action_embed(valid_actions, single_step)
        key_inputs = [valid_actions_embed]
        if not self.phase_branch:
            phase_embed = cast(nn.Module, self.phase_embed)(phase)
            phase_embed = phase_embed.unsqueeze(-2).expand(
                valid_actions_embed.shape[:-1] + (phase_embed.shape[-1],)
            )
            key_inputs.append(phase_embed)
        key_input = torch.cat(key_inputs, dim=-1)
        key_input = self.key_layer_norm(key_input)
        if self.phase_branch:
            key = self.key_model(key_input, phase)
        else:
            key = self.key_model(key_input)

        if self.phase_branch:
            query = self.query_model(backbone_embed, phase)
        else:
            query = self.query_model(backbone_embed)
        # (..., A)
        attn_score = torch.einsum("...q,...vq->...v", query, key) / self.query_dim**0.5
        attn_score = attn_score.masked_fill(
            valid_actions[..., 0] == -1,
            -float("inf"),
        )

        if self.use_nosignal:
            # On phase == "signal", we want to upweight the "nosignal" choice
            # relative to the other choices based on what trick we're on and
            # how many other tricks there are.

            # (...)
            signal_phase = phase == self.signal_phase
            # (...)
            n_valid_actions = (valid_actions[..., 0] != -1).sum(dim=-1).float()
            p_signal = cast(Tensor, self.p_signal)[trick.long()]
            eps = torch.tensor(1e-5)
            no_signal_wgt = torch.log(
                torch.maximum((1 - p_signal) / p_signal * (n_valid_actions - 1), eps)
            )
            no_signal_wgt = torch.where(n_valid_actions == 1, 0.0, no_signal_wgt)

            # The nosignal is at the 0th index.
            attn_score[..., 0] += torch.where(signal_phase, no_signal_wgt, 0.0)

        log_probs = self.log_softmax(attn_score)

        return log_probs


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
    def __init__(self, input_dim: int, num_cards: int, num_players: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_cards * num_players)
        self.num_players = num_players
        self.num_cards = num_cards

    def forward(
        self,
        backbone_embed: Tensor,
    ) -> tuple[Tensor]:
        return self.fc(backbone_embed).view(
            backbone_embed.shape[:-1] + (self.num_cards, self.num_players)
        )


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
            "hist": self.backbone_model.hist_model.hist_params,
            "mlp": self.backbone_model.mlp_params,
        }

    def start_single_step(self):
        self.backbone_model.hist_model.start_single_step()

    def stop_single_step(self):
        self.backbone_model.hist_model.stop_single_step()

    def get_state(self):
        return self.backbone_model.hist_model.state

    def set_state(self, state):
        self.backbone_model.hist_model.state = state

    def forward(self, inps):
        backbone_embed = self.backbone_model(inps["hist"], inps["private"])
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
        embed_models["action"],
        embed_models["turn"],
        embed_models["phase"],
        hp.embed_dim,
        hp.hist_hidden_dim,
        hp.hist_num_layers,
        hp.hist_output_dim,
        hp.hist_dropout,
        hp.tasks_embed_dim,
        hp.hist_phase_branch,
        settings,
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
        hp.backbone_phase_branch,
        settings,
    )
    if hp.aux_info_coef:
        aux_info_head = AuxInfoHead(
            input_dim=backbone_model.output_dim,
            num_cards=settings.num_cards,
            num_players=settings.num_players,
        )
    else:
        aux_info_head = None

    policy_head = PolicyHead(
        backbone_model.output_dim,
        cast(ActionModel, embed_models["action"]),
        cast(PaddedEmbed, embed_models["phase"]),
        hp.policy_hidden_dim,
        hp.policy_num_hidden_layers,
        hp.policy_dropout,
        hp.policy_query_dim,
        hp.policy_signal_prior,
        hp.policy_phase_branch,
        settings,
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


def load_model_for_eval(path: Path):
    settings_dict = torch.load(path / "settings.pth", weights_only=False)
    hp = settings_dict["hp"]
    settings = settings_dict["settings"]
    models = get_models(hp, settings)
    pv_model = cast(PolicyValueModel, models["pv"])
    checkpoint = torch.load(
        path / "checkpoint.pth",
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    pv_model.load_state_dict(checkpoint["pv_model"])
    pv_model.eval()
    pv_model.start_single_step()
    return pv_model, settings
