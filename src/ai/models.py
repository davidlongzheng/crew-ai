from typing import cast

import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..game.settings import Settings
from .embedding import HandModel, HandsModel, get_embed_models
from .hyperparams import Hyperparams
from .utils import MLP


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
        concat_inputs: bool,
    ):
        super().__init__()
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.card_embed = card_embed
        self.turn_embed = turn_embed
        self.concat_inputs = concat_inputs

        input_dim = 4 * embed_dim if concat_inputs else embed_dim
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
        player_embed = self.player_embed(hist_inps["player_idxs"])
        trick_embed = self.trick_embed(hist_inps["tricks"])
        card_embed = self.card_embed(hist_inps["cards"])
        turn_embed = self.turn_embed(hist_inps["turns"])
        if self.concat_inputs:
            x = torch.cat([player_embed, trick_embed, card_embed, turn_embed], dim=-1)
        else:
            x = player_embed + trick_embed + card_embed + turn_embed
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
        network_type: str,
        hist_model: HistoryModel,
        hand_embed: HandModel | HandsModel,
        player_embed: nn.Module,
        trick_embed: nn.Module,
        turn_embed: nn.Module,
        embed_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        dropout: float,
        use_layer_norm: bool,
        concat_inputs: bool,
        use_resid: bool,
    ):
        super().__init__()
        self.network_type = network_type
        self.hist_model = hist_model
        self.hand_embed = hand_embed
        self.concat_inputs = concat_inputs
        self.player_embed = player_embed
        self.trick_embed = trick_embed
        self.turn_embed = turn_embed
        self.output_dim = output_dim
        input_dim = (
            self.hist_model.output_dim
            + (embed_dim * 3 if concat_inputs else embed_dim)
            + self.hand_embed.output_dim
        )
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else None
        self.mlp = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            num_hidden_layers,
            dropout=dropout,
            use_resid=use_resid,
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
        hand_embed: Tensor = self.hand_embed(
            private_inps["hands" if self.network_type == "value" else "hand"]
        )
        player_embed = self.player_embed(private_inps["player_idx"])
        trick_embed = self.trick_embed(private_inps["trick"])
        turn_embed = self.turn_embed(private_inps["turn"])
        if self.concat_inputs:
            private_embed = torch.cat([player_embed, trick_embed, turn_embed], dim=-1)
        else:
            private_embed = player_embed + trick_embed + turn_embed

        inps = [hist_embed, private_embed, hand_embed]
        x = torch.cat(inps, dim=-1)
        if self.layer_norm:
            x = self.layer_norm(x)

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


class ValueModel(nn.Module):
    def __init__(
        self,
        backbone_model: BackboneModel,
        value_head: ValueHead,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.value_head = value_head

    def forward(self, inps):
        return self.value_head(
            self.backbone_model(inps["hist"], inps["private"], inps["seq_lengths"])
        )


def get_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    ret: dict[str, nn.Module] = {}
    for network_type in ["policy", "value"]:
        embed_models = get_embed_models(hp, settings, network_type=network_type)
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
            hp.hist_concat_inputs,
        )
        backbone_model = BackboneModel(
            network_type,
            hist_model,
            cast(HandsModel, embed_models["hands"])
            if network_type == "value"
            else cast(HandModel, embed_models["hand"]),
            embed_models["player"],
            embed_models["trick"],
            embed_models["turn"],
            hp.embed_dim,
            hp.backbone_hidden_dim,
            hp.backbone_num_hidden_layers,
            hp.backbone_output_dim,
            hp.backbone_dropout,
            hp.backbone_use_layer_norm,
            hp.backbone_concat_inputs,
            hp.backbone_use_resid,
        )
        if network_type == "policy":
            policy_head = PolicyHead(
                input_dim=hp.backbone_output_dim,
                card_embed=embed_models["card"],
                embed_dim=hp.embed_dim,
                query_dim=hp.policy_query_dim,
            )
            ret["policy"] = PolicyModel(
                backbone_model,
                policy_head,
            )
        else:
            value_head = ValueHead(
                input_dim=hp.backbone_output_dim,
            )
            ret["value"] = ValueModel(backbone_model, value_head)
    return ret
