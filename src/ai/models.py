import torch
from torch import nn
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

    def reset_state(self):
        self.state = None

    def forward(self, hist_inps, seq_lengths=None):
        assert (seq_lengths is not None) == (len(hist_inps["player_idxs"].shape) == 2)
        x = (
            self.player_embed(hist_inps["player_idxs"])
            + self.trick_embed(hist_inps["tricks"])
            + self.card_embed(hist_inps["cards"])
            + self.turn_embed(hist_inps["turns"])
        )
        use_state = len(x.shape) == 1
        if use_state:
            assert not self.training

        if not use_state and self.state is not None:
            raise Exception(
                "Forgot to reset_state() before going back to batched mode."
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        if seq_lengths is not None:
            x = pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=False
            )
        elif use_state:
            x = x.unsqueeze(dim=0)

        x, state = self.lstm(x, self.state)

        if seq_lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        elif use_state:
            x = x.squeeze(dim=0)

        # Only save state if we are using the model one action
        # at a time.
        if use_state:
            self.state = state

        x = self.fc(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DecisionModel(nn.Module):
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

    def reset_state(self):
        self.hist_model.reset_state()

    def forward(
        self,
        hist_inps,
        private_inps,
        seq_lengths=None,
    ):
        hist_embed = self.hist_model(
            hist_inps=hist_inps,
            seq_lengths=seq_lengths,
        )
        private_inp = (
            self.hand_embed(private_inps["hand"])
            + self.player_embed(private_inps["player_idx"])
            + self.trick_embed(private_inps["trick"])
            + self.turn_embed(private_inps["turn"])
        )

        x = torch.cat([hist_embed, private_inp], axis=-1)
        x = self.mlp(x)
        return x


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        decision_model: DecisionModel,
        card_embed: nn.Module,
        embed_dim: int,
        query_dim: int,
    ):
        super().__init__()
        # Decision backbone
        self.decision_model = decision_model

        # Policy Head
        self.card_embed = card_embed
        self.query_model = nn.Linear(self.decision_model.output_dim, query_dim)
        self.key_model = nn.Linear(embed_dim, query_dim)
        self.query_dim = query_dim

        # Value Head
        self.fc = nn.Linear(self.decision_model.output_dim, 1)

    def reset_state(self):
        self.decision_model.reset_state()

    def forward(
        self,
        inps,
    ):
        decision_embed = self.decision_model(
            inps["hist"],
            inps["private"],
            seq_lengths=inps["seq_lengths"],
        )
        valid_actions_embed = self.card_embed(inps["valid_actions"])

        query = self.query_model(decision_embed)
        key = self.key_model(valid_actions_embed)
        attn_score = torch.einsum("...q,...vq->...v", query, key) / self.query_dim**0.5
        attn_score = attn_score.masked_fill(
            inps["valid_actions"][..., 0] == -1,
            -float("inf"),
        )
        probs = nn.functional.softmax(attn_score, dim=-1)

        value = self.fc(decision_embed).squeeze(-1)

        return probs, value


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
    decision_model = DecisionModel(
        hist_model,
        embed_models["hand"],
        embed_models["player"],
        embed_models["trick"],
        embed_models["turn"],
        hp.embed_dim,
        hp.decision_hidden_dim,
        hp.decision_num_hidden_layers,
        hp.decision_output_dim,
        hp.decision_dropout,
        hp.decision_use_layer_norm,
    )
    pv_model = PolicyValueModel(
        decision_model,
        embed_models["card"],
        embed_dim=hp.embed_dim,
        query_dim=hp.policy_query_dim,
    )
    return embed_models | {
        "hist": hist_model,
        "decision": decision_model,
        "pv": pv_model,
    }
