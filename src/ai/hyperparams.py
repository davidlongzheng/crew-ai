import dataclasses
import re
from dataclasses import dataclass

import click
import torch

from ..lib.utils import coerce_string


@dataclass(frozen=True, kw_only=True)
class Hyperparams:
    float_dtype: torch.dtype = torch.float32
    use_mixed_precision: bool = False
    # Use torch.profile
    use_profile: bool = False
    # Profile memory at key points.
    profile_memory: bool = False

    # Each round is a single iteration of PPO clip.
    num_rounds: int = 100
    # Each epoch goes through all the trajectories of a given round.
    num_epochs_per_round: int = 3
    # Number of trajectories in an epoch.
    num_train_rollouts_per_round: int = 8192
    num_val_rollouts_per_round: int = 1024
    # Number of trajectories in a batch.
    batch_size: int = 160
    lr: float = 0.000624
    lr_schedule: str = "constant"
    lr_min_frac: float = 0.8
    lr_warmup_frac: float = 0.05
    weight_decay: float = 0.002225

    # Adam betas
    beta_1: float = 0.9
    beta_2: float = 0.999
    early_stop_num_epochs: int | None = None
    policy_early_stop_max_kl: float = 0.10
    grad_norm_clip: float = 0.3716

    # For advantage calculation
    # How much to discount future TDs in GAE calculation.
    # gae_lambda=1 converges to advantage = future_rewards - V(s_t)
    # gae_lambda=0 converges to advantage = r_t + V(s_{t+1}) - V(s_t)
    gae_lambda: float = 0.9355

    # For loss function calculations
    policy_ppo_clip_ratio: float = 0.2
    policy_ppo_coef: float = 1.0
    policy_entropy_coef: float = 1e-2
    value_loss_method: str = "mse"
    value_smooth_l1_beta: float = 0.25
    value_coef: float = 1.0
    aux_info_coef: float = 0.0

    # For embeddings
    embed_dim: int = 32
    embed_dropout: float = 0.04286
    # thermometer encodings for
    # player, trick, rank
    embed_use_pos: bool = False
    embed_sep_trump_rank: bool = True

    # For embedding a hand from a set of card embeddings
    hand_hidden_dim: int = 64
    hand_embed_dim: int = 48
    hand_num_hidden_layers: int = 1
    hand_use_layer_norm: bool = True
    hand_dropout: float = 0.04286
    hand_agg_method: str = "maxpool"

    # For tasks embedding
    tasks_hidden_dim: int = 64
    tasks_embed_dim: int = 48
    tasks_num_hidden_layers: int = 1
    tasks_use_layer_norm: bool = True
    tasks_dropout: float = 0.04286
    tasks_agg_method: str = "maxpool"

    # For public history LSTM/Transformer
    hist_hidden_dim: int = 160
    hist_output_dim: int = 144
    hist_num_layers: int = 1
    hist_dropout: float = 0.04286
    hist_phase_branch: bool = False

    # For backbone MLP
    backbone_hidden_dim: int = 512
    backbone_num_hidden_layers: int = 2
    backbone_output_dim: int = 16
    backbone_dropout: float = 0.04286
    backbone_no_head: bool = False

    # For policy network
    policy_hidden_dim: int = 96
    policy_num_hidden_layers: int = 1
    policy_dropout: float = 0.04286
    policy_query_dim: int = 80  # Attention vector on policy output.
    policy_signal_prior: str = "lin"
    policy_phase_branch: bool = False


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
            ("agg_method", r"^.*agg_method$"),
        ]:
            if batch_alias not in kwargs:
                continue

            for field in fields:
                if re.match(batch_regex, field.name) and field.name not in kwargs:
                    kwargs[field.name] = kwargs[batch_alias]
            kwargs.pop(batch_alias)

        if "float_dtype" in kwargs:
            kwargs["float_dtype"] = getattr(torch, kwargs["float_dtype"])

        return Hyperparams(**kwargs)


HP_TYPE = HyperparamsType()
