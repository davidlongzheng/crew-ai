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
    value_coef: float = 1.0
    entropy_coef: float = 1e-4
    early_stop_kl: float = 0.05

    # For embeddings
    embed_dim: int = 16
    embed_dropout: float = 0.1

    # For embedding a hand from a set of card embeddings
    hand_hidden_dim: int = 32
    hand_num_hidden_layers: int = 1
    hand_use_layer_norm: bool = True
    hand_dropout: float = 0.1

    # For public history LSTM
    hist_hidden_dim: int = 16
    hist_output_dim: int = 16
    hist_num_layers: int = 1
    hist_use_layer_norm: bool = True
    hist_dropout: float = 0.1

    # For backbone MLP
    backbone_hidden_dim: int = 16
    backbone_num_hidden_layers: int = 2
    backbone_output_dim: int = 16
    backbone_dropout: float = 0.1
    backbone_use_layer_norm: bool = False

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

        if "float_dtype" in kwargs:
            kwargs["float_dtype"] = getattr(torch, "float_dtype")

        return Hyperparams(**kwargs)


HP_TYPE = HyperparamsType()
