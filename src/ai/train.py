from __future__ import absolute_import

import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import cast

import click
import numpy as np
import pandas as pd
import torch
from ipdb import launch_ipdb_on_exception
from loguru import logger
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import cpp_game
from ai.curriculum import Curriculum, get_curriculum
from ai.hyperparams import HP_TYPE, Hyperparams
from ai.models import PolicyValueModel, get_models
from ai.rollout import do_batch_rollout_cpp
from ai.summary_writer import CustomSummaryWriter
from ai.utils import (
    Timer,
    create_lr_sched,
    create_optim,
    get_device,
    get_phase_weights,
    num_params,
    print_memory_usage,
    should_keep_outdir,
    win_rate_by_difficulty,
)
from game.settings import DEFAULT_PRESET, SETTINGS_TYPE, Settings, get_preset


@cache
def gae_advantage_discount(num_moves, gae_lambda, device, dtype):
    i = torch.arange(num_moves).view(-1, 1)
    j = torch.arange(num_moves).view(1, -1)
    exponent = i - j
    mask = exponent >= 0
    return torch.where(
        mask, gae_lambda**exponent, torch.zeros_like(exponent, dtype=dtype)
    ).to(device)


def compute_advantage(
    td: TensorDict,
    gae_lambda: float,
    pv_model: PolicyValueModel,
    device: torch.device,
    hp: Hyperparams,
):
    N, T = td["inps", "private", "player_idx"].shape
    with torch.no_grad():
        batch_size = hp.batch_size * 4
        values = torch.empty(N, T, device=device)

        for i in range(0, N, batch_size):
            eff_batch_size = min(batch_size, N - i)
            _, batch_values, _ = pv_model(td["inps"][i : i + eff_batch_size])
            values[i : i + eff_batch_size] = batch_values

    td["orig_values"] = values
    T = values.shape[-1]
    fut_values = torch.roll(values, shifts=-1, dims=-1)
    fut_values[..., -1] = 0.0
    # resids here refers to Bellman TD residuals.
    resids = td["rewards"] + fut_values - values
    advantages = resids @ gae_advantage_discount(T, gae_lambda, device, resids.dtype)
    value_targets = td["rewards"].flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    # (N, T)
    # More than one valid action
    mask = (td["inps", "valid_actions"][..., 0] != -1).sum(dim=-1) > 1
    # (N*T)
    mask = mask.reshape((-1,))
    masked_advantage = advantages.reshape((-1,))[mask]
    advantage_mean, advantage_std = masked_advantage.mean(), masked_advantage.std()
    norm_advantages = (advantages - advantage_mean) / advantage_std

    td["unnorm_advantages"] = advantages
    td["advantages"] = norm_advantages
    td["value_targets"] = value_targets


@cache
def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()


def compute_aux_info_loss(td):
    # (B, C)
    targets = td["aux_info"].long()
    # (B, T, C, P)
    pred = td["aux_info_preds"]
    _, T, _, P = pred.shape

    # (B, T, C)
    targets = targets.unsqueeze(-2).expand(-1, T, -1)
    targets = targets.reshape(-1)
    pred = pred.reshape(-1, P)
    loss = get_cross_entropy_loss()(pred, targets)

    return loss


def compute_loss(
    td,
    ppo_clip_ratio,
    ppo_coef,
    entropy_coef,
    value_coef,
    aux_info_coef,
    phase_weights,
):
    # (N, T)
    actions = td["actions"]
    # (N, T, H)
    orig_log_probs = td["orig_log_probs"]
    log_probs = td["log_probs"]
    # (N, T)
    advantages = td["advantages"]

    # (N, T)
    orig_action_log_probs = torch.gather(
        orig_log_probs, dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    # (N, T)
    action_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)
    # (N, T)
    # Mask selects for > 1 valid action
    mask = (td["inps", "valid_actions"][..., 0] != -1).sum(dim=-1) > 1
    # (N*T)
    mask = mask.reshape((-1,))

    def weight_by_phase(x):
        if phase_weights is None:
            return x
        return phase_weights * x

    def masked_mean(x):
        # (N*T)
        x = x.reshape((-1,))
        return torch.mean(x[mask])

    def g(advantages):
        return torch.where(
            advantages >= 0,
            (1 + ppo_clip_ratio) * advantages,
            (1 - ppo_clip_ratio) * advantages,
        )

    raw = (action_log_probs - orig_action_log_probs).exp() * advantages
    ceil = g(advantages)
    ppo_loss = masked_mean(weight_by_phase(-torch.minimum(raw, ceil)))
    frac_clipped = masked_mean((raw > ceil).float())
    eps = 1e-8
    log_eps = np.log(eps)
    clamped_probs = torch.exp(log_probs).clamp(min=eps)
    clamped_orig_probs = torch.exp(orig_log_probs).clamp(min=eps)
    clamped_log_probs = log_probs.clamp(min=log_eps)
    clamped_orig_log_probs = orig_log_probs.clamp(min=log_eps)
    entropy_loss = masked_mean(
        weight_by_phase(torch.sum(clamped_probs * clamped_log_probs, dim=-1))
    )
    kl_loss = masked_mean(
        torch.sum(
            clamped_orig_probs * (clamped_orig_log_probs - clamped_log_probs),
            dim=-1,
        )
    )

    values = td["values"]
    value_targets = td["value_targets"]
    value_loss = F.mse_loss(
        values,
        value_targets,
        weight=(
            phase_weights.unsqueeze(0).expand(values.shape)
            if phase_weights is not None
            else None
        ),
    )

    combined_loss = (
        ppo_coef * ppo_loss + entropy_coef * entropy_loss + value_coef * value_loss
    )
    if aux_info_coef:
        aux_info_loss = compute_aux_info_loss(td)
        combined_loss += aux_info_coef * aux_info_loss

    ret = {
        "loss": combined_loss,
        "ppo_loss": ppo_loss,
        "entropy_loss": entropy_loss,
        "kl_loss": kl_loss,
        "frac_clipped": frac_clipped,
        "value_loss": value_loss,
    }
    if aux_info_coef:
        ret["aux_info_loss"] = aux_info_loss

    return ret


def compute_grad_norms(losses, pv_model):
    ret = {}
    for loss_name in ["ppo", "value", "entropy", "aux_info"]:
        if f"{loss_name}_loss" not in losses:
            continue
        params = [y for x in pv_model.param_groups.values() for y in x]
        grads = torch.autograd.grad(
            losses[f"{loss_name}_loss"],
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        assert len(params) == len(grads)

        i = 0
        for group_name, param_group in pv_model.param_groups.items():
            grads_group = grads[i : i + len(param_group)]
            g_norms = [g.norm(2) for g in grads_group if g is not None]
            if g_norms:
                ret[f"{loss_name}_{group_name}"] = torch.norm(
                    torch.stack(g_norms), 2
                ).item()
            i += len(param_group)

    return ret


def train_one_epoch(
    round,
    epoch,
    mode,
    data_loader,
    hp,
    optim,
    pv_model,
    outdir,
    gc,
    phase_weights,
):
    running_losses = Counter()
    running_grad_norms = Counter()
    num_steps_in_epoch = 0
    num_grad_norms = 0

    if round == 0 and epoch == 0 and mode == "train" and hp.use_profile:
        prof = torch.profiler.profile(
            activities=([ProfilerActivity.CPU, ProfilerActivity.CUDA]),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(outdir)),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,  # optional
        )
    else:
        prof = None

    with prof if prof else nullcontext():
        num_batches_per_epoch = len(data_loader)
        for batch_idx, td_batch in enumerate(data_loader):
            if prof is not None:
                prof.step()

            if mode == "train":
                optim.zero_grad()

            with torch.no_grad() if mode == "val" else nullcontext():
                (
                    td_batch["log_probs"],
                    td_batch["values"],
                    td_batch["aux_info_preds"],
                ) = pv_model(td_batch["inps"])

                losses = compute_loss(
                    td_batch,
                    hp.policy_ppo_clip_ratio,
                    hp.policy_ppo_coef,
                    hp.policy_entropy_coef,
                    hp.value_coef,
                    hp.aux_info_coef,
                    phase_weights,
                )

            if mode == "train":
                # Skip round == 0 for this calculation to avoid interacting with
                # use_profile=True.
                if (
                    (not prof or round > 0)
                    and epoch == 0
                    and batch_idx < 0.05 * num_batches_per_epoch
                ):
                    grad_norms = compute_grad_norms(losses, pv_model)
                    for k, v in grad_norms.items():
                        running_grad_norms[k] += v
                    num_grad_norms += 1

                losses["loss"].backward()

                if hp.grad_norm_clip:
                    total_norm = clip_grad_norm_(
                        pv_model.parameters(), hp.grad_norm_clip
                    )
                    if total_norm.item() >= hp.grad_norm_clip:
                        gc["num_grad_clips"] += 1

                optim.step()

            for k, v in losses.items():
                running_losses[k] += v.item()

            num_steps_in_epoch += 1

    if prof:
        for k in ["self_cpu_time_total", "self_cuda_time_total"]:
            print(f"By {k}")
            print(prof.key_averages().table(sort_by=k, row_limit=10))

    losses = {k: v / num_steps_in_epoch for k, v in running_losses.items()}

    if num_grad_norms > 0:
        grad_norms = {k: v / num_grad_norms for k, v in running_grad_norms.items()}
    else:
        grad_norms = None

    return (
        losses,
        num_steps_in_epoch,
        grad_norms,
    )


def train_one_round(
    round,
    pv_model,
    td,
    hp,
    optim,
    lr_sched,
    gc,
    writer,
    log_epoch,
    outdir,
    phase_weights,
):
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

    best_loss_epoch, best_loss = None, float("inf")
    writer.add_scalar("lr/lr", lr_sched.get_last_lr()[0], round)
    policy_kl = None
    for epoch in range(hp.num_epochs_per_round):
        if log_epoch:
            logger.info(f"Epoch {epoch}")
        for mode in ["train", "val"]:
            if mode == "train":
                pv_model.train()
                data_loader = train_data_loader
            else:
                pv_model.eval()
                data_loader = val_data_loader
            (
                losses,
                num_steps_in_epoch,
                grad_norms,
            ) = train_one_epoch(
                round=round,
                epoch=epoch,
                mode=mode,
                data_loader=data_loader,
                hp=hp,
                optim=optim,
                pv_model=pv_model,
                outdir=outdir,
                gc=gc,
                phase_weights=phase_weights,
            )
            if mode == "train":
                gc["num_steps"] += num_steps_in_epoch
                gc["num_epochs"] += 1

            if grad_norms:
                for k, v in grad_norms.items():
                    writer.add_scalar(f"grad_norms/{k}", v, gc["num_epochs"])

            for k, v in losses.items():
                if k == "loss":
                    writer.add_scalar(
                        f"main_loss/loss_{mode}",
                        v,
                        gc["num_epochs"],
                    )
                else:
                    writer.add_scalar(
                        f"aux_loss/loss_{k}_{mode}",
                        v,
                        gc["num_epochs"],
                    )

            if mode == "train":
                gc["num_rollout_passes"] += hp.num_train_rollouts_per_round
                writer.add_scalar(
                    "counter/round",
                    round,
                    gc["num_epochs"],
                )
                for k in [
                    "num_rollout_passes",
                    "num_rollouts",
                    "num_steps",
                    "num_early_stops",
                    "num_early_stop_max_kls",
                    "num_grad_clips",
                    "num_resets",
                ]:
                    writer.add_scalar(
                        f"counter/{k}",
                        gc[k],
                        gc["num_epochs"],
                    )

            if mode == "train":
                policy_kl = losses["kl_loss"]

            if mode == "val" and losses["loss"] < best_loss:
                best_loss_epoch, best_loss = epoch, losses["loss"]

        if (
            hp.early_stop_num_epochs
            and epoch - best_loss_epoch >= hp.early_stop_num_epochs
        ):
            gc["num_early_stops"] += 1
            logger.debug(f"Early stopping at epoch {epoch}")
            break

        if policy_kl > hp.policy_early_stop_max_kl:
            gc["num_early_stop_max_kls"] += 1
            logger.debug(f"Max KL reached at epoch {epoch}. Early stopping.")
            break

        (outdir / "_keep").touch()

    lr_sched.step()
    gc["num_rollouts"] += hp.num_train_rollouts_per_round


def save_task_info(
    td: TensorDict,
    round: int,
    outdir: Path,
):
    (outdir / "task_infos").mkdir(exist_ok=True)
    # compute num samples and success rate per task_idx + game difficulty.
    # compute num samples and win rate per game difficulty.

    df = pd.DataFrame(
        {
            "task_idx": td["task_idxs_no_pt"].cpu().numpy().tolist(),
            "task_success": td["task_success"].cpu().numpy().tolist(),
            "game_win": td["win"].cpu().numpy(),
            "game_difficulty": td["difficulty"].cpu().numpy(),
        }
    )
    df["round"] = round
    df["count"] = 1

    task_df = df.explode(["task_idx", "task_success"])
    task_df = task_df.query("task_idx != -1")
    task_df = task_df.groupby(["round", "task_idx", "game_difficulty"])[
        ["task_success", "count"]
    ].sum()

    game_df = df.groupby(["round", "game_difficulty"])[["game_win", "count"]].sum()
    ret = {
        "task": task_df,
        "game": game_df,
    }
    out_fn = outdir / "task_infos" / f"task_info_{round}.pkl"
    pd.to_pickle(ret, str(out_fn))


def train(
    device: torch.device,
    pv_model: PolicyValueModel,
    optim: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    settings: Settings,
    hp: Hyperparams,
    writer: SummaryWriter,
    outdir: Path,
    start_round: int,
    gc: Counter,
    rng_state,
    num_threads: int,
    skip_td: bool,
    save_task_info_freq: int,
    curriculum: Curriculum | None,
    curriculum_state: dict,
    last_win_rate: float,
) -> None:
    timer = Timer(writer)

    if curriculum:
        settings = cast(
            Settings, curriculum.update_settings(curriculum_state, settings)
        )
        logger.info(f"New settings: {settings}")

    cpp_settings = settings.to_cpp()
    batch_rollout = cpp_game.BatchRollout(
        cpp_settings,
        hp.num_train_rollouts_per_round + hp.num_val_rollouts_per_round,
        num_threads=num_threads,
    )
    phase_weights = get_phase_weights(settings, hp, device)

    if rng_state is not None:
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state_all(rng_state[1])

    round = start_round
    while round < hp.num_rounds:
        torch.save(
            {
                "round": round,
                "curriculum_state": curriculum_state,
                "pv_model": pv_model.state_dict(),
                "optim": optim.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "gc": gc,
                "rng_state": (torch.get_rng_state(), torch.cuda.get_rng_state_all()),
                "last_win_rate": last_win_rate,
            },
            outdir / "checkpoint_unverified.pth",
        )

        start_time = time.time()
        pv_model.eval()

        if hp.profile_memory:
            print_memory_usage("before rollout")

        timer.start("rollout")
        seed = int(torch.randint(0, 1_000_000_000, ()))
        td = do_batch_rollout_cpp(
            batch_rollout,
            batch_seed=seed,
            pv_model=pv_model,
            device=device,
        )
        timer.finish("rollout", round)

        if hp.profile_memory:
            print_memory_usage("after rollout")

        win_rate = td["win"].float().mean()
        if win_rate < last_win_rate - hp.reset_thresh:
            logger.warning(
                f"Win rate {win_rate:.3f} is less than {last_win_rate:.3f} by more than {hp.reset_thresh:.3f}. Resetting model."
            )
            state_dict = torch.load(outdir / "checkpoint.pth", weights_only=False)
            pv_model.load_state_dict(state_dict["pv_model"])
            optim.load_state_dict(state_dict["optim"])
            lr_sched.load_state_dict(state_dict["lr_sched"])
            gc["num_resets"] += 1
            continue

        (outdir / "checkpoint_unverified.pth").replace(outdir / "checkpoint.pth")

        if curriculum and (
            new_settings := curriculum.update_settings(curriculum_state, settings, td)
        ):
            settings = new_settings
            logger.info(f"New settings: {settings}")
            cpp_settings = settings.to_cpp()
            batch_rollout = cpp_game.BatchRollout(
                cpp_settings,
                hp.num_train_rollouts_per_round + hp.num_val_rollouts_per_round,
                num_threads=num_threads,
            )
            last_win_rate = 0.0
            continue

        if save_task_info_freq and round % save_task_info_freq == 0:
            save_task_info(td, round, outdir)

        if not skip_td:
            torch.save(td, outdir / "td.pth")

        mean_reward = td["rewards"].sum(dim=-1).mean()
        writer.add_scalar("rewards/reward_mean", mean_reward, round)
        mean_frac_success = td["task_success"].float().mean(dim=1).mean()
        assert mean_frac_success <= 1.0
        writer.add_scalar("rewards/frac_success_mean", mean_frac_success, round)
        writer.add_scalar("rewards/win_rate", win_rate, round)
        writer.add_scalar("rewards/win_rate_vs_rollouts", win_rate, gc["num_rollouts"])

        win_rates = win_rate_by_difficulty(td)
        for diff, _win_rate in win_rates.items():
            writer.add_scalar(f"win_rate_by_diff/win_rate_{diff}", _win_rate, round)

        writer.add_hparams(
            hparam_dict={
                k: v
                for k, v in asdict(hp).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            metric_dict={
                "metrics/reward_mean": mean_reward,
                "metrics/frac_success": mean_frac_success,
                "metrics/win_rate": win_rate,
            },
            global_step=round,
        )

        timer.start("advantage")
        compute_advantage(td, hp.gae_lambda, pv_model, device, hp)
        for k in ["orig_value", "value_target", "unnorm_advantage"]:
            writer.add_scalar(f"rewards/{k}_mean", td[f"{k}s"].mean(), round)
            writer.add_scalar(f"rewards/{k}_std", td[f"{k}s"].std(), round)
        timer.finish("advantage", round)

        if hp.profile_memory:
            print_memory_usage("after advantage")

        timer.start("training")
        train_one_round(
            round=round,
            pv_model=pv_model,
            td=td,
            hp=hp,
            optim=optim,
            lr_sched=lr_sched,
            gc=gc,
            writer=writer,
            log_epoch=(hp.num_rounds <= 5),
            outdir=outdir,
            phase_weights=phase_weights,
        )
        timer.finish("training", round)

        if hp.profile_memory:
            print_memory_usage("after train")
            return

        elapsed = time.time() - start_time
        logger.info(
            f"Round {round}: mean_reward={mean_reward:.3f}, win_rate={win_rate:.3f}, elapsed={elapsed:.3f}s"
        )

        last_win_rate = win_rate
        round += 1


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
    default=get_preset(DEFAULT_PRESET),
    help="Settings",
)
@click.option(
    "--curriculum-mode",
    help="Curriculum mode",
)
@click.option("--device", type=torch.device, default=get_device(), help="Device")
@click.option(
    "--seed",
    type=int,
    help="Seed",
    default=43,
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
    "--copy-dir",
    type=Path,
    help="Copy contents of a different outdir",
)
@click.option(
    "--no-error-catch",
    is_flag=True,
    help="Don't catch errors",
)
@click.option(
    "--num-threads",
    type=int,
    default=1,
    help="Number of rollout threads",
)
@click.option(
    "--skip-td",
    is_flag=True,
    help="Skip save td.",
)
@click.option(
    "--save-task-info-freq", type=int, default=10, help="Save task info every n rounds."
)
@click.option(
    "--skip-load-win-rate",
    is_flag=True,
    help="Skip load win rate.",
)
def main(
    outdir: Path,
    hp: Hyperparams,
    settings: Settings,
    curriculum_mode: str | None,
    device: torch.device,
    seed: int,
    clean: bool,
    resume: bool,
    copy_dir: Path | None,
    no_error_catch: bool,
    num_threads: int,
    skip_td: bool,
    save_task_info_freq: int,
    skip_load_win_rate: bool,
) -> None:
    outdir = outdir.resolve()
    autoindex_runs = not (
        outdir.name.startswith("run_") or outdir.name.startswith("seed_")
    )

    if autoindex_runs:
        max_run_idx = max(
            [
                int(x.name.split("_")[-1])
                for x in outdir.glob("run_*")
                if should_keep_outdir(x)
            ],
            default=-1,
        )
        if clean or resume:
            outdir = outdir / f"run_{max(max_run_idx, 0)}"
        else:
            outdir = outdir / f"run_{max_run_idx + 1}"

    if outdir.exists():
        if clean or not should_keep_outdir(outdir):
            logger.info(f"** Cleaning outdir {outdir} **")
            shutil.rmtree(outdir)
        elif not resume:
            raise Exception("Must set --clean or --resume to run on existing outdir.")

    if copy_dir and not outdir.exists():
        logger.info(f"Copying contents from {copy_dir} to {outdir}")
        shutil.copytree(copy_dir, outdir)
        for bname in ["_running", "_keep"]:
            if (outdir / bname).exists():
                (outdir / bname).unlink()

    outdir.mkdir(parents=True, exist_ok=True)
    logger.add(outdir / "train.log")
    assert not (outdir / "_running").exists()
    (outdir / "_running").touch()

    try:
        logger.info("** Training Configuration **")
        logger.info(f"Settings: {settings}")
        if curriculum_mode is not None:
            curriculum = get_curriculum(curriculum_mode)
            logger.info(f"Curriculum: {curriculum_mode}")
        else:
            curriculum = None
        logger.info(f"Hyperparams: {hp}")
        logger.info(f"Output Directory: {outdir}")
        logger.info(f"Device: {device}")
        logger.info(f"Training Seed: {seed}")
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(seed)

        if hp.profile_memory:
            print_memory_usage("before create models")

        logger.info("** Creating models, optimizer **")
        models = get_models(hp, settings)
        for m in models.values():
            m.to(device)
        pv_model = cast(PolicyValueModel, models["pv"])
        optim = create_optim([pv_model], hp)
        lr_sched = create_lr_sched(optim, hp)

        if (outdir / "checkpoint.pth").exists():
            logger.info("Loading state from checkpoint")
            orig_state_dict = torch.load(outdir / "checkpoint.pth", weights_only=False)
        else:
            orig_state_dict = {}

        for obj, key in [
            (pv_model, "pv_model"),
            (optim, "optim"),
            (lr_sched, "lr_sched"),
        ]:
            if key in orig_state_dict and obj is not None:
                obj.load_state_dict(orig_state_dict[key])
                del orig_state_dict[key]

        logger.info(f"Num Parameters: pv={num_params(pv_model):.2e}")

        torch.save(
            {
                "outdir": outdir,
                "settings": settings,
                "curriculum": curriculum,
                "hp": hp,
                "seed": seed,
                "device": device,
            },
            outdir / "settings.pth",
        )

        logger.info("** Training **")
        with CustomSummaryWriter(str(outdir)) as writer:
            with launch_ipdb_on_exception() if not no_error_catch else nullcontext():
                train(
                    device,
                    pv_model,
                    optim,
                    lr_sched,
                    settings,
                    hp,
                    writer,
                    outdir,
                    orig_state_dict.get("round", 0),
                    orig_state_dict.get("gc", Counter()),
                    orig_state_dict.get("rng_state", None),
                    num_threads,
                    skip_td,
                    save_task_info_freq,
                    curriculum,
                    orig_state_dict.get("curriculum_state", {}),
                    0.0
                    if skip_load_win_rate
                    else orig_state_dict.get("last_win_rate", 0.0),
                )
    finally:
        (outdir / "_running").unlink()


if __name__ == "__main__":
    main()
