from __future__ import absolute_import

import shutil
from pathlib import Path
from typing import Any, cast

import click
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tensordict.nn import set_composite_lp_aggregate
from torch.nn.utils.rnn import pad_sequence
from torchrl.envs import check_env_specs

from ..game.settings import Settings, easy_settings
from ..lib.types import StrMap
from .environment import get_torchrl_env
from .featurizer import featurize
from .hyperparams import Hyperparams
from .models import PolicyValueModel, get_models
from .rollout import do_rollout


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else 0
        if torch.cuda.is_available()
        else "cpu"
    )


def create_optim(model, lr: float, weight_decay: float):
    named_params = list(model.named_parameters())
    no_wd_params = [
        p for name, p in named_params if "bias" in name or "layer_norm" in name
    ]
    no_wd_params_set = set(no_wd_params)
    other_params = [p for _, p in named_params if p not in no_wd_params_set]

    param_groups: list[dict[str, Any]] = [
        {"params": other_params},
        {"params": no_wd_params, "weight_decay": 0},
    ]
    optimizer = optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def gae_discount_matrix(n, factor):
    i = torch.arange(n).view(-1, 1)
    j = torch.arange(n).view(1, -1)
    exponent = i - j
    mask = exponent >= 0
    return torch.where(
        mask, factor**exponent, torch.zeros_like(exponent, dtype=torch.float)
    )


def compute_advantage(
    rollouts: list[StrMap],
    gae_gamma: float,
    gae_lambda: float,
    pv_model: PolicyValueModel,
    device: torch.device,
    batch_size: int,
    settings: Settings,
):
    for idx in range(0, len(rollouts), batch_size):
        rollouts_batch = rollouts[idx : idx + batch_size]
        inps = featurize(
            public_history=[x["public_history"] for x in rollouts_batch],
            private_inputs=[x["private_inputs"] for x in rollouts_batch],
            valid_actions=[x["valid_actions"] for x in rollouts_batch],
            non_feature_dims=2,
            settings=settings,
            device=device,
        )
        with torch.no_grad():
            probs, values = pv_model(inps)
        probs = probs.to("cpu")
        values = values.to("cpu")

        for rollout, _probs, _values in zip(rollouts_batch, probs, values):
            rollout["orig_probs"] = _probs
            seq_length = len(rollout["private_inputs"])

            advantage = torch.zeros(seq_length)
            value_targets = torch.zeros(seq_length)

            for player_idx in range(settings.num_players):
                player_final_reward = rollout["rewards_pp"][player_idx]
                # player made decisions on the following idxs.
                player_move_idxs = torch.tensor(
                    [
                        i
                        for i, x in enumerate(rollout["private_inputs"])
                        if x["player_idx"] == player_idx
                    ]
                )
                n = len(player_move_idxs)
                player_values = torch.index_select(
                    _values, dim=0, index=player_move_idxs
                )
                fut_player_values = gae_gamma * torch.roll(player_values, -1)
                fut_player_values[-1] = rollout["rewards_pp"][player_idx]
                player_tds = fut_player_values - player_values
                player_advantage = (
                    gae_discount_matrix(n, gae_gamma * gae_lambda) @ player_tds
                )
                player_value_targets = (
                    gae_gamma ** torch.arange(n - 1, -1, -1) * player_final_reward
                )
                advantage[player_move_idxs] = player_advantage
                value_targets[player_move_idxs] = player_value_targets

            rollout["advantage"] = advantage
            rollout["value_targets"] = value_targets


def get_loss(
    probs,
    values,
    targets,
    orig_probs,
    advantages,
    value_targets,
    mse_criterion,
    kl_criterion,
    ppo_clip_ratio,
    value_coef,
    entropy_coef,
):
    orig_action_probs = torch.index_select(orig_probs, dim=-1, index=targets)
    action_probs = torch.index_select(
        probs,
        dim=-1,
        index=targets,
    )

    def g(advantages):
        torch.where(
            advantages >= 0,
            (1 + ppo_clip_ratio) * advantages,
            (1 - ppo_clip_ratio) * advantages,
        )

    policy_loss = -torch.minimum(
        action_probs / orig_action_probs * advantages, g(advantages)
    )
    value_loss = value_coef * mse_criterion(values, value_targets)
    entropy_loss = entropy_coef * torch.mean(
        torch.sum(action_probs * torch.log(action_probs), dim=-1)
    )
    kl_loss = kl_criterion(probs, orig_probs)
    combined_loss = policy_loss + value_loss + entropy_loss
    return combined_loss, policy_loss, value_loss, entropy_loss, kl_loss


def train(
    device: torch.device,
    pv_model: PolicyValueModel,
    optimizer: optim.Optimizer,
    settings: Settings,
    hp: Hyperparams,
) -> None:
    mse_criterion = nn.MSELoss()
    kl_criterion = nn.KLDivLoss()

    def pad_seq(inp, *, dtype=None):
        return pad_sequence(
            [torch.tensor(x, dtype=dtype, device=device) for x in inp],
            batch_first=True,
        )

    for epoch in range(hp.num_epochs):
        pv_model.eval()
        rollouts: list[StrMap] = []
        for _ in range(hp.num_rollouts_per_epoch):
            seed = int(torch.randint(0, 100000, ()))
            rollout = do_rollout(
                settings,
                seed=seed,
                pv_model=pv_model,
            )
            rollouts.append(rollout)

        compute_advantage(
            rollouts,
            hp.gae_gamma,
            hp.gae_lambda,
            pv_model,
            device,
            hp.batch_size,
            settings,
        )

        pv_model.train()

        for idx in range(0, len(rollouts), hp.batch_size):
            rollouts_batch = rollouts[idx : idx + hp.batch_size]
            inps = featurize(
                public_history=[x["public_history"] for x in rollouts_batch],
                private_inputs=[x["private_inputs"] for x in rollouts_batch],
                valid_actions=[x["valid_actions"] for x in rollouts_batch],
                non_feature_dims=2,
                settings=settings,
                device=device,
            )
            targets = pad_seq([x["targets"] for x in rollouts_batch])
            orig_probs = pad_seq([x["orig_probs"] for x in rollouts_batch])
            advantages = pad_seq([x["advantage"] for x in rollouts_batch])
            value_targets = pad_seq([x["value_targets"] for x in rollouts_batch])

            optimizer.zero_grad()
            probs, values = pv_model(inps)
            combined_loss, policy_loss, value_loss, entropy_loss, kl_loss = get_loss(
                probs,
                values,
                targets,
                orig_probs,
                advantages,
                value_targets,
                mse_criterion,
                kl_criterion,
                hp.ppo_clip_ratio,
                hp.value_coef,
                hp.entropy_coef,
            )
            if kl_loss >= hp.early_stop_kl:
                break
            combined_loss.backward()
            optimizer.step()


@click.command()
@click.option(
    "--outdir",
    type=Path,
    help="Outdir",
    required=True,
)
@click.option(
    "--hp",
    type=Hyperparams.parse_str,
    default=Hyperparams(),
    help="Hyperparams",
)
@click.option(
    "--settings",
    type=Settings.parse_str,
    default=easy_settings(),
    help="Hyperparams",
)
@click.option(
    "--seed",
    type=int,
    help="Seed",
    default=42,
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clean outdir",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Must set --resume to run on existing outdir.",
)
def main(
    outdir: Path,
    hp: Hyperparams,
    settings: Settings,
    seed: int,
    clean: bool,
    resume: bool,
) -> None:
    outdir = outdir.resolve()
    if clean:
        logger.info(f"** Cleaning outdir {outdir} **")
        shutil.rmtree(outdir)
    if not resume and outdir.exists():
        raise Exception("Must set --clean or --resume to run on existing outdir.")
    outdir.mkdir(exist_ok=True)
    logger.add(outdir / "train.log")

    logger.info("** Training Configuration **")
    logger.info(f"Settings: {settings}")
    logger.info(f"Hyperparams: {hp}")
    logger.info(f"Output Directory: {outdir}")
    torch.set_default_dtype(hp.float_dtype)

    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Training Seed: {seed}")
    torch.manual_seed(seed)

    logger.info("** Creating models, optimizer **")
    pv_model: PolicyValueModel = cast(PolicyValueModel, get_models(hp, settings)["pv"])
    pv_model = pv_model.to(device)
    optimizer = create_optim(pv_model, hp.lr, hp.weight_decay)
    env = get_torchrl_env(settings)
    check_env_specs(env)
    # writer = SummaryWriter(str(outdir))
    set_composite_lp_aggregate(False).set()

    logger.info("** Training **")
    train(
        device,
        pv_model,
        optimizer,
        settings,
        hp,
    )


if __name__ == "__main__":
    main()
