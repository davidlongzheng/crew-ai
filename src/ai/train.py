from __future__ import absolute_import

import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict
from torch import GradScaler, nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..game.settings import SETTINGS_TYPE, Settings, get_preset
from .aux_info import get_aux_info_spec, set_aux_info_hist_only
from .featurizer import featurize
from .hyperparams import HP_TYPE, Hyperparams
from .models import PolicyValueModel, get_models
from .rollout import do_batch_rollout
from .summary_writer import CustomSummaryWriter
from .win_cache import CANON_WIN_CACHE_DIR, load_cache


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    return torch.device(0 if torch.cuda.is_available() else "cpu")


def should_keep_outdir(x):
    return (x / "_running").exists() or (x / "_keep").exists()


def create_optim(models, hp: Hyperparams):
    named_params = []
    seen_params = set()
    for model in models:
        for name, p in model.named_parameters():
            if p in seen_params:
                continue
            named_params.append((name, p))
            seen_params.add(p)
    no_wd_params = [
        p for name, p in named_params if "bias" in name or "layer_norm" in name
    ]
    no_wd_params_set = set(no_wd_params)
    other_params = [p for _, p in named_params if p not in no_wd_params_set]

    param_groups: list[dict[str, Any]] = [
        {"params": other_params},
        {"params": no_wd_params, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        betas=(hp.beta_1, hp.beta_2),
    )
    return optimizer


def create_lr_sched(optimizer, hp):
    if hp.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp.num_rounds,
            eta_min=hp.lr * hp.lr_min_frac,
        )
    elif hp.lr_schedule == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: 1 - (x / (hp.num_rounds - 1)) * (1 - hp.lr_min_frac)
        )
    else:
        assert hp.lr_schedule == "constant", hp.lr_schedule
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
        )


@cache
def gae_advantage_discount(num_moves, gae_lambda, device, dtype):
    i = torch.arange(num_moves).view(-1, 1)
    j = torch.arange(num_moves).view(1, -1)
    exponent = i - j
    mask = exponent >= 0
    return torch.where(
        mask, gae_lambda**exponent, torch.zeros_like(exponent, dtype=dtype)
    ).to(device)


class Timer:
    def __init__(self, writer: SummaryWriter, log_first_n=3):
        self.writer = writer
        self.times: dict[str, float | None] = {}
        self.log_first_n = log_first_n
        self.num_logged: defaultdict = defaultdict(lambda: 0)

    def start(self, key):
        assert self.times.get(key) is None
        self.times[key] = time.time()

    def finish(self, key, global_step):
        assert self.times.get(key) is not None
        elapsed = time.time() - self.times[key]
        self.writer.add_scalar(f"times/{key}_time", elapsed, global_step)
        if self.num_logged[key] < self.log_first_n:
            logger.info(f"{key}_time: {elapsed:.3f}s")
        self.num_logged[key] += 1
        self.times[key] = None


def compute_advantage(
    td: TensorDict,
    gae_lambda: float,
    pv_model: PolicyValueModel,
    device: torch.device,
):
    with torch.no_grad():
        _, td["orig_values"], _ = pv_model(td["inps"])

    values = td["orig_values"]
    T = values.shape[-1]
    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == T).all()
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
def get_mse_loss():
    return nn.MSELoss()


@cache
def get_huber_loss():
    return nn.HuberLoss()


@cache
def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()


@cache
def get_smooth_l1_loss(beta):
    return nn.SmoothL1Loss(beta=beta)


def compute_aux_info_loss(td, spec, separate=False):
    num_cont = spec["num_cat"].isna().sum()
    assert spec["num_cat"].head(num_cont).isna().all(), (
        "Continuous vars must be at the beginning of the spec."
    )

    targets = td["aux_infos"]
    preds = td["aux_info_preds"]
    weights = torch.tensor(spec["weight"], device=targets.device)
    if separate:
        losses = []
    else:
        loss = 0

    if num_cont:
        cont_targets = targets[..., :num_cont]
        cont_preds = preds[..., :num_cont]
        cont_weights = weights[:num_cont]
        cont_loss = cont_weights * (cont_preds - cont_targets) ** 2
        if separate:
            losses += torch.mean(cont_loss, dim=(0, 1)).tolist()
        else:
            loss += torch.mean(cont_loss)

    num_cats = spec["num_cat"].iloc[num_cont:].astype(int)
    cat_weights = weights[num_cont:]
    cat_targets = targets[..., num_cont:].long()
    cat_preds = preds[..., num_cont:]
    cur_pred_idx = 0
    for i, (cat_weight, num_cat) in enumerate(zip(cat_weights, num_cats)):
        cat_target = cat_targets[..., i]
        cat_pred = cat_preds[..., cur_pred_idx : cur_pred_idx + num_cat]
        cat_loss = cat_weight * get_cross_entropy_loss()(
            cat_pred.reshape((-1, cat_pred.shape[-1])),
            cat_target.reshape((-1,)),
        )
        cur_pred_idx += num_cat
        if separate:
            losses.append(cat_loss.item())
        else:
            loss += cat_loss

    if separate:
        return losses
    else:
        return loss


def compute_loss(
    td,
    aux_info_spec,
    ppo_clip_ratio,
    ppo_coef,
    entropy_coef,
    value_loss_method,
    value_smooth_l1_beta,
    value_coef,
    aux_info_coef,
):
    # (N, T)
    actions = td["actions"]
    # (N, T, H)
    orig_probs = td["orig_probs"]
    probs = td["probs"]
    orig_log_probs = td["orig_log_probs"]
    log_probs = td["log_probs"]
    # (N, T)
    advantages = td["advantages"]

    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == advantages.shape[1]).all()

    # (N, T)
    orig_action_probs = torch.gather(
        orig_probs, dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    # (N, T)
    action_probs = torch.gather(
        probs,
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)
    # (N, T)
    # At least one valid action
    mask = (td["inps", "valid_actions"][..., 0] != -1).sum(dim=-1) > 1
    # (N*T)
    mask = mask.reshape((-1,))

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

    raw = (action_probs / orig_action_probs) * advantages
    ceil = g(advantages)
    ppo_loss = masked_mean(-torch.minimum(raw, ceil))
    frac_clipped = masked_mean((raw > ceil).float())
    eps = 1e-8
    log_eps = np.log(eps)
    clamped_probs = probs.clamp(min=eps)
    clamped_orig_probs = orig_probs.clamp(min=eps)
    clamped_log_probs = log_probs.clamp(min=log_eps)
    clamped_orig_log_probs = orig_log_probs.clamp(min=log_eps)
    entropy_loss = masked_mean(torch.sum(clamped_probs * clamped_log_probs, dim=-1))
    kl_loss = masked_mean(
        torch.sum(
            clamped_orig_probs * (clamped_orig_log_probs - clamped_log_probs),
            dim=-1,
        )
    )

    def get_value_loss(x, y):
        if value_loss_method == "mse":
            return get_mse_loss()(x, y)
        elif value_loss_method == "huber":
            return get_huber_loss()(x, y)
        elif value_loss_method == "smoothl1":
            return get_smooth_l1_loss(value_smooth_l1_beta)(x, y)
        else:
            raise ValueError(value_loss_method)

    values = td["values"]
    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == values.shape[1]).all()
    value_targets = td["value_targets"]
    value_loss = get_value_loss(values, value_targets)

    combined_loss = (
        ppo_coef * ppo_loss + entropy_coef * entropy_loss + value_coef * value_loss
    )
    if aux_info_coef:
        aux_info_loss = compute_aux_info_loss(
            td,
            aux_info_spec,
        )
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
    for loss_name in ["ppo", "value", "entropy"]:
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


def get_train_data(
    hp: Hyperparams,
    settings: Settings,
    device: torch.device,
    pv_model: PolicyValueModel,
    timer: Timer,
    round: int,
    engine_seeds: list[int] | None,
):
    def pad_seq(inp, *, dtype=None):
        return pad_sequence(
            [torch.tensor(x, dtype=dtype, device=device) for x in inp],
            batch_first=True,
        )

    timer.start("rollout")
    seed = int(torch.randint(0, 100_000_000, ()))
    rollouts = do_batch_rollout(
        settings,
        num_rollouts=hp.num_train_rollouts_per_round + hp.num_val_rollouts_per_round,
        batch_seed=seed,
        engine_seeds=engine_seeds,
        pv_model=pv_model,
        device=device,
    )
    timer.finish("rollout", round)

    timer.start("featurize")
    inps = featurize(
        public_history=[x["public_history"] for x in rollouts],
        private_inputs=[x["private_inputs"] for x in rollouts],
        valid_actions=[x["valid_actions"] for x in rollouts],
        task_idxs=[x["task_idxs"] for x in rollouts],
        non_feature_dims=2,
        settings=settings,
        device=device,
    )
    actions = pad_seq([x["actions"] for x in rollouts])
    orig_probs = pad_seq([x["probs"] for x in rollouts])
    orig_log_probs = pad_seq([x["log_probs"] for x in rollouts])
    rewards = pad_seq([x["rewards"] for x in rollouts])
    frac_success = torch.tensor(
        [
            np.sum([y[0] for y in x["num_success_tasks_pp"]])
            / np.sum([y[1] for y in x["num_success_tasks_pp"]])
            for x in rollouts
        ],
        device=device,
    )
    win = torch.tensor(
        [x["win"] for x in rollouts],
        device=device,
    )
    aux_infos = pad_seq([x["aux_infos"] for x in rollouts])

    td = TensorDict(
        inps=inps,
        actions=actions,
        orig_probs=orig_probs,
        orig_log_probs=orig_log_probs,
        rewards=rewards,
        frac_success=frac_success,
        win=win,
        aux_infos=aux_infos,
    )
    td.auto_batch_size_()
    timer.finish("featurize", round)

    return td


def train_one_epoch(
    round,
    epoch,
    mode,
    data_loader,
    hp,
    optim,
    pv_model,
    aux_info_spec,
    outdir,
    scaler,
):
    running_losses = Counter()
    running_grad_norms = Counter()
    num_steps_in_epoch = 0
    num_grad_norms = 0

    if round == 0 and epoch == 0 and mode == "train" and hp.use_profile:
        prof = torch.profiler.profile(
            activities=(
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if scaler
                else [ProfilerActivity.CPU]
            ),
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=3),
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
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if scaler
                    else nullcontext()
                ):
                    (
                        (td_batch["probs"], td_batch["log_probs"]),
                        td_batch["values"],
                        td_batch["aux_info_preds"],
                    ) = pv_model(td_batch["inps"])

                    losses = compute_loss(
                        td_batch,
                        aux_info_spec,
                        hp.policy_ppo_clip_ratio,
                        hp.policy_ppo_coef,
                        hp.policy_entropy_coef,
                        hp.value_loss_method,
                        hp.value_smooth_l1_beta,
                        hp.value_coef,
                        hp.aux_info_coef,
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

                if scaler:
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optim)
                else:
                    losses["loss"].backward()

                if hp.grad_norm_clip:
                    clip_grad_norm_(pv_model.parameters(), hp.grad_norm_clip)

                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()

            for k, v in losses.items():
                running_losses[k] += v.item()

            num_steps_in_epoch += 1

    if prof:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))

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
    aux_info_spec,
    scaler,
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
                aux_info_spec=aux_info_spec,
                outdir=outdir,
                scaler=scaler,
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
                gc["num_rollouts"] += hp.num_train_rollouts_per_round
                writer.add_scalar(
                    "counter/num_rollouts",
                    gc["num_rollouts"],
                    gc["num_epochs"],
                )
                writer.add_scalar(
                    "counter/round",
                    round,
                    gc["num_epochs"],
                )
                writer.add_scalar(
                    "counter/num_steps",
                    gc["num_steps"],
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
            logger.debug(f"Early stopping at epoch {epoch}")
            break

        if policy_kl > hp.policy_early_stop_max_kl:
            logger.debug(f"Max KL reached at epoch {epoch}. Early stopping.")
            break

        (outdir / "_keep").touch()

    lr_sched.step()


def train(
    device: torch.device,
    pv_model: PolicyValueModel,
    optim: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    settings: Settings,
    hp: Hyperparams,
    writer: SummaryWriter,
    outdir: Path,
    start_round: int,
    gc: Counter,
    engine_seeds: list[int] | None,
) -> None:
    timer = Timer(writer)
    aux_info_spec = get_aux_info_spec(settings)

    for round in range(start_round, hp.num_rounds):
        logger.info(f"Round {round}")
        pv_model.eval()

        td = get_train_data(
            hp=hp,
            settings=settings,
            device=device,
            pv_model=pv_model,
            timer=timer,
            round=round,
            engine_seeds=engine_seeds,
        )

        mean_reward = td["rewards"].sum(dim=-1).mean()
        writer.add_scalar("rewards/reward_mean", mean_reward, round)
        mean_frac_success = td["frac_success"].mean()
        win_rate = td["win"].float().mean()
        assert mean_frac_success <= 1.0
        writer.add_scalar("rewards/frac_success_mean", mean_frac_success, round)
        writer.add_scalar("rewards/win_rate", win_rate, round)

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
        compute_advantage(td, hp.gae_lambda, pv_model, device)
        for k in ["orig_value", "value_target", "unnorm_advantage"]:
            writer.add_scalar(f"rewards/{k}_mean", td[f"{k}s"].mean(), round)
            writer.add_scalar(f"rewards/{k}_std", td[f"{k}s"].std(), round)
        timer.finish("advantage", round)

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
            aux_info_spec=aux_info_spec,
            scaler=scaler,
        )
        timer.finish("training", round)

        torch.save(
            {
                "round": round + 1,
                "td": td,
                "pv_model": pv_model.state_dict(),
                "optim": optim.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "gc": gc,
                "scaler": (scaler.state_dict() if scaler else None),
            },
            outdir / "checkpoint.pth",
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
    default=get_preset("easy_p4"),
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
    "--autoindex-runs",
    is_flag=True,
    help="Auto-index new runs.",
)
@click.option(
    "--copy-dir",
    type=Path,
    help="Copy contents of a different outdir",
)
@click.option(
    "--win-thresh",
    type=float,
    default=None,
    help="Use win cache to only choose engine seeds above a certain threshold.",
)
@click.option(
    "--win-cache-dir", type=Path, default=CANON_WIN_CACHE_DIR, help="Win cache dir."
)
def main(
    outdir: Path,
    hp: Hyperparams,
    settings: Settings,
    device: torch.device,
    seed: int,
    clean: bool,
    resume: bool,
    autoindex_runs: bool,
    copy_dir: Path | None,
    win_thresh: float | None,
    win_cache_dir: Path,
) -> None:
    outdir = outdir.resolve()
    # nocommit a bit hacky
    assert autoindex_runs == ("run" not in outdir.name), (
        "We expect outdir to be named with run_* unless autoindex_runs=True"
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
    (outdir / "_running").touch()

    if (outdir / "checkpoint.pth").exists():
        logger.info("Loading state from checkpoint")
        orig_state_dict = torch.load(outdir / "checkpoint.pth", weights_only=False)
    else:
        orig_state_dict = {}

    try:
        logger.info("** Training Configuration **")
        logger.info(f"Settings: {settings}")
        logger.info(f"Hyperparams: {hp}")
        logger.info(f"Output Directory: {outdir}")
        logger.info(f"Device: {device}")
        logger.info(f"Training Seed: {seed}")
        torch.set_default_dtype(hp.float_dtype)
        torch.manual_seed(seed)

        logger.info("** Creating models, optimizer **")
        if hp.aux_info_hist_only:
            set_aux_info_hist_only(True)

        models = get_models(hp, settings)
        for m in models.values():
            m.to(device)
        pv_model = cast(PolicyValueModel, models["pv"])
        optim = create_optim([pv_model], hp)
        lr_sched = create_lr_sched(optim, hp)
        scaler = GradScaler() if device.type == "cuda" else None
        for obj, key in [
            (pv_model, "pv_model"),
            (optim, "optim"),
            (lr_sched, "lr_sched"),
            (scaler, "scaler"),
        ]:
            if key in orig_state_dict:
                obj.load_state_dict(orig_state_dict[key])
                del orig_state_dict[key]

        logger.info(f"Num Parameters: pv={num_params(pv_model):.2e}")

        if win_thresh is not None:
            win_df = load_cache(settings, win_cache_dir)
            engine_seeds = list(win_df[win_df["win_rate"] >= win_thresh]["seed"].values)
        else:
            engine_seeds = None

        torch.save(
            {
                "outdir": outdir,
                "settings": settings,
                "hp": hp,
                "seed": seed,
                "device": device,
            },
            outdir / "settings.pth",
        )

        logger.info("** Training **")
        with CustomSummaryWriter(str(outdir)) as writer:
            train(
                device,
                pv_model,
                optim,
                lr_sched,
                scaler,
                settings,
                hp,
                writer,
                outdir,
                orig_state_dict.get("round", 0),
                orig_state_dict.get("gc", Counter()),
                engine_seeds,
            )
    finally:
        (outdir / "_running").unlink()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
