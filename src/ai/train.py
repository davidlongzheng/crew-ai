from __future__ import absolute_import

import functools
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..game.settings import SETTINGS_TYPE, Settings, easy_settings
from .featurizer import featurize
from .hyperparams import HP_TYPE, Hyperparams
from .models import PolicyModel, get_models, make_td_modules
from .rollout import do_batch_rollout
from .summary_writer import CustomSummaryWriter


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else 0
        if torch.cuda.is_available()
        else "cpu"
    )


def create_optim(module, lr: float, weight_decay: float):
    named_params = list(module.named_parameters())
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


@functools.lru_cache
def gae_advantage_discount(num_moves, gae_gamma, gae_lambda, device, dtype):
    i = torch.arange(num_moves).view(-1, 1)
    j = torch.arange(num_moves).view(1, -1)
    exponent = i - j
    mask = exponent >= 0
    factor = gae_gamma * gae_lambda
    return torch.where(
        mask, factor**exponent, torch.zeros_like(exponent, dtype=dtype)
    ).to(device)


@functools.lru_cache
def gae_gamma_discount(num_moves, gae_gamma, device):
    return gae_gamma ** torch.arange(num_moves - 1, -1, -1).to(device)


def flat_to_pp_tensor(values, player_idxs):
    """Convert a tensor of shape (N, T) to a shape of (N, P, M) using
    a player_idx tensor of shape (N, T).
    where P = number of players, and M = most number of moves made by
    a single player.
    """
    N, T = values.shape
    P = player_idxs.max().item() + 1
    assert T % P == 0
    M = T // P

    player_idxs = player_idxs.type(torch.LongTensor)

    # Let's assume no padding for now and not handle it.
    assert (player_idxs >= 0).all()

    # Shape (N, T, P)
    one_hot = F.one_hot(player_idxs, num_classes=P)

    # Shape (N, T, P)
    cumsum = one_hot.cumsum(dim=1)

    # Shape (N, T)
    # Gets which player move M for a given (N, T)
    ranks = (cumsum * one_hot).sum(dim=-1) - 1

    pp_values = torch.zeros(N, P, M, device=values.device, dtype=values.dtype)
    n_idx = torch.arange(N, device=values.device)[:, None].expand(N, T).reshape(-1)
    p_idx = player_idxs.reshape(-1)
    m_idx = ranks.reshape(-1)
    pp_values[n_idx, p_idx, m_idx] = values.reshape(-1)

    return pp_values


def pp_to_flat_tensor(pp_values, player_idxs):
    """Reverse of flat_to_pp_tensor()"""
    # Let's assume no padding for now and not handle it.
    assert (player_idxs >= 0).all()
    N, P, M = pp_values.shape
    _, T = player_idxs.shape
    assert P * M == T

    player_idxs = player_idxs.type(torch.LongTensor)

    # (N, T, P)
    one_hot = F.one_hot(player_idxs, num_classes=P)
    # (N, T, P)
    cumsum = one_hot.cumsum(dim=1)
    ranks = (cumsum * one_hot).sum(dim=-1) - 1

    n_idx = torch.arange(N, device=pp_values.device)[:, None].expand(N, T).reshape(-1)
    p_idx = player_idxs.reshape(-1)
    r_idx = ranks.reshape(-1)

    values = pp_values[n_idx, p_idx, r_idx].reshape(N, T)
    return values


def compute_advantage(
    td: TensorDict,
    gae_gamma: float,
    gae_lambda: float,
    value_module: TensorDictModule,
):
    with torch.no_grad():
        value_module(td)

    player_idxs = td["inps"]["private"]["player_idx"]
    pp_values = flat_to_pp_tensor(td["values"], player_idxs)
    num_moves = pp_values.shape[2]
    fut_pp_values = gae_gamma * torch.roll(pp_values, shifts=-1, dims=-1)
    fut_pp_values[..., -1] = td["rewards"]
    # resids here refers to Bellman TD residuals.
    pp_resids = fut_pp_values - pp_values
    pp_advantages = pp_resids @ gae_advantage_discount(
        num_moves, gae_gamma, gae_lambda, td.device, pp_resids.dtype
    )
    pp_value_targets = torch.einsum(
        "np,m->npm", td["rewards"], gae_gamma_discount(num_moves, gae_gamma, td.device)
    )
    advantages = pp_to_flat_tensor(pp_advantages, player_idxs)
    std_advantages = (advantages - advantages.mean()) / advantages.std()

    td["unnorm_advantages"] = advantages
    td["advantages"] = std_advantages
    td["value_targets"] = pp_to_flat_tensor(pp_value_targets, player_idxs)


def compute_loss(
    td,
    mse_criterion,
    ppo_clip_ratio,
    value_coef,
    entropy_coef,
):
    actions = td["actions"]
    values = td["values"]
    orig_log_probs = td["orig_log_probs"]
    log_probs = td["log_probs"]
    advantages = td["advantages"]
    value_targets = td["value_targets"]

    orig_action_log_probs = torch.gather(
        orig_log_probs, dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    action_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)

    def g(advantages):
        return torch.where(
            advantages >= 0,
            (1 + ppo_clip_ratio) * advantages,
            (1 - ppo_clip_ratio) * advantages,
        )

    policy_loss = torch.mean(
        -torch.minimum(
            torch.exp(action_log_probs - orig_action_log_probs) * advantages,
            g(advantages),
        )
    )
    value_loss = value_coef * mse_criterion(values, value_targets)
    eps = np.log(1e-8)
    clamped_log_probs = log_probs.clamp(min=eps)
    clamped_orig_log_probs = orig_log_probs.clamp(min=eps)
    entropy_loss = entropy_coef * torch.mean(
        torch.sum(torch.exp(clamped_log_probs) * clamped_log_probs, dim=-1)
    )
    kl_loss = torch.mean(
        torch.sum(
            torch.exp(clamped_orig_log_probs)
            * (clamped_orig_log_probs - clamped_log_probs),
            dim=-1,
        )
    )
    combined_loss = policy_loss + value_loss + entropy_loss

    return combined_loss, policy_loss, value_loss, entropy_loss, kl_loss


class Timer:
    def __init__(self, writer: SummaryWriter, log_first=False):
        self.writer = writer
        self.times: dict[str, float | None] = {}
        self.log_first = log_first
        self.has_logged: set[str] = set()

    def start(self, key):
        assert self.times.get(key) is None
        self.times[key] = time.time()

    def finish(self, key, global_step):
        assert self.times.get(key) is not None
        elapsed = time.time() - self.times[key]
        self.writer.add_scalar(f"times/{key}_time", elapsed, global_step)
        if self.log_first and key not in self.has_logged:
            logger.info(f"{key}_time: {elapsed:.3}s")
            self.has_logged.add(key)
        self.times[key] = None


def train_one_epoch(
    mode,
    data_loader,
    hp,
    optimizer,
    pv_module,
    mse_criterion,
):
    (
        running_combined_loss,
        running_policy_loss,
        running_value_loss,
        running_entropy_loss,
        running_kl_loss,
    ) = 0, 0, 0, 0, 0
    num_steps_in_epoch = 0
    for td_batch in data_loader:
        if mode == "train":
            optimizer.zero_grad()
            pv_module(td_batch)
        else:
            with torch.no_grad():
                pv_module(td_batch)
        combined_loss, policy_loss, value_loss, entropy_loss, kl_loss = compute_loss(
            td_batch,
            mse_criterion,
            hp.ppo_clip_ratio,
            hp.value_coef,
            hp.entropy_coef,
        )
        if mode == "train":
            combined_loss.backward()
            optimizer.step()
        running_combined_loss += combined_loss.item()
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
        running_entropy_loss += entropy_loss.item()
        running_kl_loss += kl_loss.item()
        num_steps_in_epoch += 1
    final_combined_loss = running_combined_loss / num_steps_in_epoch
    final_policy_loss = running_policy_loss / num_steps_in_epoch
    final_value_loss = running_value_loss / num_steps_in_epoch
    final_entropy_loss = running_entropy_loss / num_steps_in_epoch
    final_kl_loss = running_kl_loss / num_steps_in_epoch

    return (
        final_combined_loss,
        final_policy_loss,
        final_value_loss,
        final_entropy_loss,
        final_kl_loss,
        num_steps_in_epoch,
    )


def get_train_data(
    hp: Hyperparams,
    settings: Settings,
    device: torch.device,
    policy_model: PolicyModel,
    timer: Timer,
    round: int,
):
    def pad_seq(inp, *, dtype=None):
        return pad_sequence(
            [torch.tensor(x, dtype=dtype, device=device) for x in inp],
            batch_first=True,
        )

    timer.start("rollout")
    seed = int(torch.randint(0, 100000, ()))
    rollouts = do_batch_rollout(
        settings,
        num_rollouts=hp.num_train_rollouts_per_round + hp.num_val_rollouts_per_round,
        seed=seed,
        policy_model=policy_model,
        device=device,
    )
    timer.finish("rollout", round)

    timer.start("featurize")
    inps = featurize(
        public_history=[x["public_history"] for x in rollouts],
        private_inputs=[x["private_inputs"] for x in rollouts],
        valid_actions=[x["valid_actions"] for x in rollouts],
        non_feature_dims=2,
        settings=settings,
        device=device,
    )
    actions = pad_seq([x["actions"] for x in rollouts])
    orig_log_probs = pad_seq([x["log_probs"] for x in rollouts])
    rewards = torch.tensor([x["rewards_pp"] for x in rollouts], device=device)
    frac_success = torch.tensor(
        [
            np.sum([y[0] for y in x["num_success_tasks_pp"]])
            / np.sum([y[1] for y in x["num_success_tasks_pp"]])
            for x in rollouts
        ],
        device=device,
    )
    td = TensorDict(
        inps=inps,
        actions=actions,
        orig_log_probs=orig_log_probs,
        rewards=rewards,
        frac_success=frac_success,
    )
    td.auto_batch_size_()
    timer.finish("featurize", round)

    return td


def train(
    device: torch.device,
    policy_model: PolicyModel,
    value_module: TensorDictModule,
    pv_module: TensorDictModule,
    optimizer: optim.Optimizer,
    settings: Settings,
    hp: Hyperparams,
    writer: SummaryWriter,
    td0_path: Path | None,
) -> None:
    mse_criterion = nn.MSELoss()

    timer = Timer(writer, log_first=True)
    gc: Counter = Counter()
    for round in range(hp.num_rounds):
        logger.info(f"Round {round}")
        pv_module.eval()

        if round == 0 and td0_path and td0_path.exists():
            logger.info(f"Loading round=0 td from {td0_path}")
            td = torch.load(td0_path, weights_only=False)
        else:
            td = get_train_data(
                hp=hp,
                settings=settings,
                device=device,
                policy_model=policy_model,
                timer=timer,
                round=round,
            )
            if round == 0 and td0_path:
                logger.info(f"Saving round=0 td to {td0_path}")
                torch.save(td, td0_path)

        mean_reward = td["rewards"].mean()
        writer.add_scalar("rewards/reward_mean", mean_reward, round)
        mean_frac_success = td["frac_success"].mean()
        assert mean_frac_success <= 1.0
        writer.add_scalar("rewards/frac_success_mean", mean_frac_success, round)

        timer.start("advantage")
        compute_advantage(td, hp.gae_gamma, hp.gae_lambda, value_module)
        writer.add_scalar("rewards/value_mean", td["values"].mean(), round)
        writer.add_scalar("rewards/value_std", td["values"].std(), round)
        writer.add_scalar(
            "rewards/value_target_mean", td["value_targets"].mean(), round
        )
        writer.add_scalar("rewards/value_target_std", td["value_targets"].std(), round)
        writer.add_scalar(
            "rewards/unnorm_advantage_mean", td["unnorm_advantages"].mean(), round
        )
        writer.add_scalar(
            "rewards/unnorm_advantage_std", td["unnorm_advantages"].std(), round
        )
        timer.finish("advantage", round)

        timer.start("training")
        train_data_loader = DataLoader(
            td[: hp.num_train_rollouts_per_round],
            hp.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: x,
        )
        val_data_loader = DataLoader(
            td[hp.num_train_rollouts_per_round :],
            hp.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: x,
        )
        for epoch in range(hp.num_epochs_per_round):
            for mode in ["train", "val"]:
                if mode == "train":
                    pv_module.train()
                    data_loader = train_data_loader
                else:
                    pv_module.eval()
                    data_loader = val_data_loader
                (
                    combined_loss,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    kl_loss,
                    num_steps_in_epoch,
                ) = train_one_epoch(
                    mode=mode,
                    data_loader=data_loader,
                    hp=hp,
                    optimizer=optimizer,
                    pv_module=pv_module,
                    mse_criterion=mse_criterion,
                )
                if mode == "train":
                    gc["num_steps"] += num_steps_in_epoch
                    gc["num_epochs"] += 1

                sfx = "" if mode == "train" else f"_{mode}"
                writer.add_scalar(
                    f"loss/combined{sfx}", combined_loss, gc["num_epochs"]
                )
                writer.add_scalar(f"loss/policy{sfx}", policy_loss, gc["num_epochs"])
                writer.add_scalar(f"loss/value{sfx}", value_loss, gc["num_epochs"])
                writer.add_scalar(f"loss/entropy{sfx}", entropy_loss, gc["num_epochs"])
                writer.add_scalar(f"loss/kl{sfx}", kl_loss, gc["num_epochs"])
                if mode == "train":
                    gc["num_rollouts"] += hp.num_train_rollouts_per_round
                    writer.add_scalar(
                        "counter/num_rollouts",
                        gc["num_rollouts"],
                        gc["num_epochs"],
                    )
                    writer.add_scalar("counter/round", round, gc["num_epochs"])
                    writer.add_scalar(
                        "counter/num_steps", gc["num_steps"], gc["num_epochs"]
                    )
        timer.finish("training", round)

        writer.add_hparams(
            hparam_dict={
                k: v
                for k, v in asdict(hp).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            metric_dict={
                "metrics/mean_reward": mean_reward,
                "metrics/frac_success": mean_frac_success,
            },
            global_step=round,
        )


@click.command()
@click.option(
    "--outdir",
    type=Path,
    help="Outdir",
    required=True,
)
@click.option(
    "--hp",
    type=HP_TYPE,
    default=Hyperparams(),
    help="Hyperparams",
)
@click.option(
    "--settings",
    type=SETTINGS_TYPE,
    default=easy_settings(),
    help="Settings",
)
@click.option("--device", type=torch.device, default=get_device(), help="Device")
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
@click.option(
    "--td0_path",
    type=Path,
    default=None,
    help="Where to cache round=0 td for fast experimentation. kinda jank, consider deprecating in the future!",
)
def main(
    outdir: Path,
    hp: Hyperparams,
    settings: Settings,
    device: torch.device,
    seed: int,
    clean: bool,
    resume: bool,
    td0_path: Path | None,
) -> None:
    outdir = outdir.resolve()
    if td0_path:
        td0_path = td0_path.resolve()
    if clean and outdir.exists():
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
    logger.info(f"Device: {device}")
    logger.info(f"Training Seed: {seed}")
    torch.set_default_dtype(hp.float_dtype)
    torch.manual_seed(seed)

    logger.info("** Creating models, optimizer **")
    models = get_models(hp, settings)
    for m in models.values():
        m.to(device)
    policy_model = cast(PolicyModel, models["policy"])

    td_modules = make_td_modules(models)
    value_module = td_modules["value"]
    pv_module = td_modules["pv"]

    optimizer = create_optim(pv_module, hp.lr, hp.weight_decay)

    logger.info("** Training **")
    with CustomSummaryWriter(str(outdir)) as writer:
        train(
            device,
            policy_model,
            value_module,
            pv_module,
            optimizer,
            settings,
            hp,
            writer,
            td0_path,
        )


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
