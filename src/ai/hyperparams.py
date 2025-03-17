import dataclasses
import re
from dataclasses import dataclass

import click
import torch

from ..lib.utils import coerce_string


@dataclass(frozen=True, kw_only=True)
class Hyperparams:
    float_dtype: torch.dtype = torch.float32

    # Each round is a single iteration of PPO clip.
    num_rounds: int = 1
    # Each epoch goes through all the trajectories of a given round.
    num_epochs_per_round: int = 20
    # Number of trajectories in an epoch.
    num_train_rollouts_per_round: int = 8192
    num_val_rollouts_per_round: int = 1024
    # Number of trajectories in a batch.
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3

    # For advantage calculation
    # How much to discount future rewards
    gae_gamma: float = 0.99
    # How much to discount future TDs in GAE calculation.
    # gae_lambda=1 converges to advantage = future_rewards - V(s_t)
    # gae_lambda=0 converges to advantage = r_t + V(s_{t+1}) - V(s_t)
    gae_lambda: float = 0.95

    # For loss function calculations
    ppo_clip_ratio: float = 0.2
    entropy_coef: float = 1e-4
    early_stop_kl: float = 0.05
    value_loss_method: str = "mse"
    value_smooth_l1_beta: float = 0.25

    # For embeddings
    embed_dim: int = 16
    embed_dropout: float = 0.1

    # For embedding a hand from a set of card embeddings
    hand_hidden_dim: int = 32
    hand_embed_dim: int = 32
    hand_num_hidden_layers: int = 1
    hand_use_layer_norm: bool = True
    hand_dropout: float = 0.1
    hand_agg_method: str = "maxpool"

    # For embedding the global set of hands
    hands_hidden_dim: int = 64
    hands_embed_dim: int = 64
    hands_num_hidden_layers: int = 1
    hands_use_layer_norm: bool = True
    hands_dropout: float = 0.1
    hands_concat_inputs: bool = False
    hands_agg_method: str = "maxpool"

    # For public history LSTM
    hist_hidden_dim: int = 32
    hist_output_dim: int = 32
    hist_num_layers: int = 1
    hist_use_layer_norm: bool = True
    hist_dropout: float = 0.1
    hist_concat_inputs: bool = False

    # For backbone MLP
    backbone_hidden_dim: int = 32
    backbone_num_hidden_layers: int = 2
    backbone_output_dim: int = 16
    backbone_dropout: float = 0.1
    backbone_use_layer_norm: bool = True
    backbone_concat_inputs: bool = False
    # nocommit stupid
    backbone_knockout_idx: int | None = None

    # For policy-value network
    policy_query_dim: int = 16  # Attention vector on policy output.


class HyperparamsType(click.ParamType):
    name = "Hyperparams"

    def convert(self, value, param, ctx):
        if isinstance(value, Hyperparams):
            return value

        kwargs = dict([x.split("=") for x in value.split(",")])
        for k, v in kwargs.items():
            kwargs[k] = coerce_string(v)

        fields = dataclasses.fields(Hyperparams)
        for batch_alias, batch_regex in [
            ("dropout", r"^.*dropout$"),
            ("use_layer_norm", r"^.*use_layer_norm$"),
            ("concat_inputs", r"^.*concat_inputs$"),
            ("agg_method", r"^.*agg_method$"),
        ]:
            if batch_alias not in kwargs:
                continue

            for field in fields:
                if re.match(batch_regex, field.name):
                    kwargs[field.name] = kwargs[batch_alias]
            kwargs.pop(batch_alias)

        if "float_dtype" in kwargs:
            kwargs["float_dtype"] = getattr(torch, "float_dtype")

        return Hyperparams(**kwargs)


HP_TYPE = HyperparamsType()
