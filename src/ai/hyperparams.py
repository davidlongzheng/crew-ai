from dataclasses import dataclass

import torch

from ..lib.utils import coerce_string


@dataclass(frozen=True, kw_only=True)
class Hyperparams:
    float_dtype: torch.dtype = torch.float16

    # Epoch is a single iteration of PPO clip.
    num_epochs: int = 20
    # Number of trajectories in an epoch.
    num_rollouts_per_epoch: int = 1028
    # Number of trajectories in a batch.
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2

    # For advantage calculation
    # How much to discount future rewards
    gae_gamma: float = 0.99
    # How much to discount future TDs in GAE calculation.
    gae_lambda: float = 0.95

    # For loss function calculations
    ppo_clip_ratio: float = 0.2
    value_coef: float = 1
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

    # For decision MLP
    decision_hidden_dim: int = 16
    decision_num_hidden_layers: int = 2
    decision_output_dim: int = 16
    decision_dropout: float = 0.1
    decision_use_layer_norm: bool = False

    # For policy-value network
    policy_query_dim: int = 16  # Attention vector on policy output.

    @staticmethod
    def parse_str(txt):
        kwargs = dict([x.split("=") for x in txt.split(",")])
        for k, v in kwargs.items():
            kwargs[k] = coerce_string(v)

        if "float_dtype" in kwargs:
            kwargs["float_dtype"] = getattr(torch, "float_dtype")

        return Hyperparams(**kwargs)
