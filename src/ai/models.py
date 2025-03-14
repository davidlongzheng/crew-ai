import torch
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..game.settings import Settings
from .embedding import get_embed_models
from .hyperparams import Hyperparams
from .utils import make_mlp


class HistoryModel(nn.Module):
    def __init__(
        self,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        card_embed: nn.Module,
        turn_embed: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.card_embed = card_embed
        self.turn_embed = turn_embed

        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.state = None
        self.single_step = False

    def start_single_step(self):
        self.single_step = True
        self.state = None

    def stop_single_step(self):
        self.single_step = False
        self.state = None

    def forward(
        self, hist_inps: TensorDict, seq_lengths: Tensor | None = None
    ) -> Tensor:
        assert (seq_lengths is not None) == (len(hist_inps["player_idxs"].shape) == 2)
        x = (
            self.player_embed(hist_inps["player_idxs"])
            + self.trick_embed(hist_inps["tricks"])
            + self.card_embed(hist_inps["cards"])
            + self.turn_embed(hist_inps["turns"])
        )
        if self.single_step:
            assert not self.training
            assert len(x.shape) <= 2

        if not self.single_step and self.state is not None:
            raise Exception(
                "Forgot to stop_single_step() before going back to multi-step mode."
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        if seq_lengths is not None:
            x = pack_padded_sequence(
                x, seq_lengths, enforce_sorted=False, batch_first=True
            )
        elif self.single_step:
            x = x.unsqueeze(dim=-2)

        x, state = self.lstm(x, self.state)

        if seq_lengths is not None:
            x, _ = pad_packed_sequence(x, padding_value=0.0, batch_first=True)
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
        hand_embed: nn.Module,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        turn_embed: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.hist_model = hist_model
        self.hand_embed = hand_embed
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.turn_embed = turn_embed
        self.output_dim = output_dim
        self.mlp = make_mlp(
            self.hist_model.output_dim + embed_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def start_single_step(self):
        self.hist_model.start_single_step()

    def stop_single_step(self):
        self.hist_model.stop_single_step()

    def forward(
        self,
        hist_inps: TensorDict,
        private_inps: TensorDict,
        seq_lengths: Tensor | None = None,
    ) -> Tensor:
        hist_embed: Tensor = self.hist_model(
            hist_inps=hist_inps,
            seq_lengths=seq_lengths,
        )
        private_inp: Tensor = (
            self.hand_embed(private_inps["hand"])
            + self.player_embed(private_inps["player_idx"])
            + self.trick_embed(private_inps["trick"])
            + self.turn_embed(private_inps["turn"])
        )

        x = torch.cat([hist_embed, private_inp], dim=-1)
        x = self.mlp(x)
        return x


class PolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        card_embed: nn.Module,
        embed_dim: int,
        query_dim: int,
    ):
        super().__init__()
        self.card_embed = card_embed
        self.query_model = nn.Linear(input_dim, query_dim)
        self.key_model = nn.Linear(embed_dim, query_dim)
        self.query_dim = query_dim
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        backbone_embed: Tensor,
        valid_actions: Tensor,
    ) -> Tensor:
        valid_actions_embed = self.card_embed(valid_actions)

        query = self.query_model(backbone_embed)
        key = self.key_model(valid_actions_embed)
        attn_score = torch.einsum("...q,...vq->...v", query, key) / self.query_dim**0.5
        attn_score = attn_score.masked_fill(
            valid_actions[..., 0] == -1,
            -float("inf"),
        )
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


class PolicyModel(nn.Module):
    def __init__(
        self,
        backbone_model: BackboneModel,
        policy_head: PolicyHead,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.policy_head = policy_head

    def start_single_step(self):
        self.backbone_model.start_single_step()

    def stop_single_step(self):
        self.backbone_model.stop_single_step()

    def forward(self, inps):
        return self.policy_head(
            self.backbone_model(inps["hist"], inps["private"], inps["seq_lengths"]),
            inps["valid_actions"],
        )


def get_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    embed_models = get_embed_models(hp, settings)
    hist_model = HistoryModel(
        embed_models["player"],
        embed_models["trick"],
        embed_models["card"],
        embed_models["turn"],
        hp.embed_dim,
        hp.hist_hidden_dim,
        hp.hist_num_layers,
        hp.hist_output_dim,
        hp.hist_dropout,
        hp.hist_use_layer_norm,
    )
    backbone_model = BackboneModel(
        hist_model,
        embed_models["hand"],
        embed_models["player"],
        embed_models["trick"],
        embed_models["turn"],
        hp.embed_dim,
        hp.backbone_hidden_dim,
        hp.backbone_num_hidden_layers,
        hp.backbone_output_dim,
        hp.backbone_dropout,
        hp.backbone_use_layer_norm,
    )
    policy_head = PolicyHead(
        input_dim=hp.backbone_output_dim,
        card_embed=embed_models["card"],
        embed_dim=hp.embed_dim,
        query_dim=hp.policy_query_dim,
    )
    value_head = ValueHead(
        input_dim=hp.backbone_output_dim,
    )
    policy_model = PolicyModel(
        backbone_model,
        policy_head,
    )
    return embed_models | {
        "hist": hist_model,
        "backbone": backbone_model,
        "policy_head": policy_head,
        "value_head": value_head,
        "policy": policy_model,
    }


def make_td_modules(
    models: dict[str, nn.Module],
) -> TensorDictModule:
    backbone_module = TensorDictModule(
        models["backbone"],
        in_keys=[("inps", "hist"), ("inps", "private"), ("inps", "seq_lengths")],
        out_keys=["backbone_embed"],
    )
    policy_head_module = TensorDictModule(
        models["policy_head"],
        in_keys=[
            "backbone_embed",
            ("inps", "valid_actions"),
        ],
        out_keys=["log_probs"],
    )
    value_head_module = TensorDictModule(
        models["value_head"], in_keys=["backbone_embed"], out_keys=["values"]
    )
    policy_module = TensorDictSequential(backbone_module, policy_head_module)
    value_module = TensorDictSequential(backbone_module, value_head_module)
    pv_module = TensorDictSequential(
        backbone_module, policy_head_module, value_head_module
    )

    return {
        "backbone": backbone_module,
        "policy_head": policy_head_module,
        "value_head": value_head_module,
        "policy": policy_module,
        "value": value_module,
        "pv": pv_module,
    }
