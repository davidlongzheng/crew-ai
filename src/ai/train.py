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
import torch.optim as optim
from loguru import logger
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..game.settings import SETTINGS_TYPE, Settings, easy_settings
from .featurizer import featurize
from .hyperparams import HP_TYPE, Hyperparams
from .models import PolicyModel, ValueModel, get_models
from .rollout import do_batch_rollout
from .summary_writer import CustomSummaryWriter


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else 0
        if torch.cuda.is_available()
        else "cpu"
    )


def should_keep_outdir(x):
    return (x / "_running").exists() or (x / "_keep").exists()


def create_optim(models, lr: float, weight_decay: float):
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
            logger.info(f"{key}_time: {elapsed:.3f}s")
            self.has_logged.add(key)
        self.times[key] = None


def compute_advantage(
    td: TensorDict,
    gae_gamma: float,
    gae_lambda: float,
    value_model: ValueModel,
):
    with torch.no_grad():
        td["orig_values"] = value_model(td["inps"])

    values = td["orig_values"]
    T = values.shape[-1]
    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == T).all()
    fut_values = gae_gamma * torch.roll(values, shifts=-1, dims=-1)
    fut_values[..., -1] = td["rewards"]
    # resids here refers to Bellman TD residuals.
    resids = fut_values - values
    advantages = resids @ gae_advantage_discount(
        T, gae_gamma, gae_lambda, td.device, resids.dtype
    )
    value_targets = torch.einsum(
        "n,t->nt", td["rewards"], gae_gamma_discount(T, gae_gamma, td.device)
    )
    norm_advantages = (advantages - advantages.mean()) / advantages.std()

    td["unnorm_advantages"] = advantages
    td["advantages"] = norm_advantages
    td["value_targets"] = value_targets


def compute_policy_loss(
    td,
    ppo_clip_ratio,
    entropy_coef,
):
    actions = td["actions"]
    orig_log_probs = td["orig_log_probs"]
    log_probs = td["log_probs"]
    advantages = td["advantages"]

    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == advantages.shape[1]).all()

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

    ppo_loss = torch.mean(
        -torch.minimum(
            torch.exp(action_log_probs - orig_action_log_probs) * advantages,
            g(advantages),
        )
    )
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
    combined_loss = ppo_loss + entropy_loss

    return {
        "loss": combined_loss,
        "ppo_loss": ppo_loss,
        "entropy_loss": entropy_loss,
        "kl_loss": kl_loss,
    }


def compute_value_loss(
    td,
    method,
    smooth_l1_beta,
):
    def get_loss(x, y):
        if method == "mse":
            return torch.mean((x - y) ** 2)
        elif method == "smoothl1":
            return torch.mean(
                torch.where(
                    torch.abs(x - y) < smooth_l1_beta,
                    0.5 * (x - y) ** 2 / smooth_l1_beta,
                    torch.abs(x - y) - 0.5 * smooth_l1_beta,
                )
            )
        else:
            raise ValueError(method)

    values = td["values"]
    # only handle this case if it comes to that.
    assert (td["inps"]["seq_lengths"] == values.shape[1]).all()
    value_targets = td["value_targets"]
    return {
        "loss": get_loss(values, value_targets),
        "loss_t0": get_loss(values[:, 0], value_targets[:, 0]),
        "loss_t-1": get_loss(values[:, -1], value_targets[:, -1]),
    }


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
    rewards = torch.tensor([x["reward"] for x in rollouts], device=device)
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


def train_one_epoch(
    mode,
    network_type,
    data_loader,
    hp,
    optim,
    model,
):
    running_losses = Counter()
    num_steps_in_epoch = 0
    for td_batch in data_loader:
        out_key = "log_probs" if network_type == "policy" else "values"
        if mode == "train":
            optim.zero_grad()
            td_batch[out_key] = model(td_batch["inps"])
        else:
            with torch.no_grad():
                td_batch[out_key] = model(td_batch["inps"])
        if network_type == "policy":
            losses = compute_policy_loss(
                td_batch,
                hp.ppo_clip_ratio,
                hp.entropy_coef,
            )
        else:
            losses = compute_value_loss(
                td_batch,
                hp.value_loss_method,
                hp.value_smooth_l1_beta,
            )

        if mode == "train":
            losses["loss"].backward()
            optim.step()

        for k, v in losses.items():
            running_losses[k] += v.item()

        num_steps_in_epoch += 1

    losses = {}
    for k, v in running_losses.items():
        losses[k] = v / num_steps_in_epoch

    return (
        losses,
        num_steps_in_epoch,
    )


def train_one_round(
    round,
    policy_model,
    value_model,
    td,
    hp,
    policy_optim,
    value_optim,
    gc,
    writer,
    log_epoch,
    outdir,
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
    for network_type in ["value"]:  # nocommit
        logger.info(f"Training {network_type} network")
        model = policy_model if network_type == "policy" else value_model
        optim = policy_optim if network_type == "policy" else value_optim
        for epoch in range(hp.num_epochs_per_round):
            if log_epoch:
                logger.info(f"Epoch {epoch}")
            for mode in ["train", "val"]:
                if mode == "train":
                    model.train()
                    data_loader = train_data_loader
                else:
                    model.eval()
                    data_loader = val_data_loader
                (
                    losses,
                    num_steps_in_epoch,
                ) = train_one_epoch(
                    mode=mode,
                    network_type=network_type,
                    data_loader=data_loader,
                    hp=hp,
                    optim=optim,
                    model=model,
                )
                if mode == "train":
                    gc[f"{network_type}_num_steps"] += num_steps_in_epoch
                    gc[f"{network_type}_num_epochs"] += 1

                for k, v in losses.items():
                    writer.add_scalar(
                        f"loss/{network_type}_{k}_{mode}",
                        v,
                        gc[f"{network_type}_num_epochs"],
                    )

                if mode == "train":
                    gc[f"{network_type}_num_rollouts"] += (
                        hp.num_train_rollouts_per_round
                    )
                    writer.add_scalar(
                        "counter/num_rollouts",
                        gc[f"{network_type}_num_rollouts"],
                        gc[f"{network_type}_num_epochs"],
                    )
                    writer.add_scalar(
                        "counter/round", round, gc[f"{network_type}_num_epochs"]
                    )
                    writer.add_scalar(
                        "counter/num_steps",
                        gc[f"{network_type}_num_steps"],
                        gc[f"{network_type}_num_epochs"],
                    )

            (outdir / "_keep").touch()


def train(
    device: torch.device,
    policy_model: PolicyModel,
    value_model: ValueModel,
    policy_optim: optim.Optimizer,
    value_optim: optim.Optimizer,
    settings: Settings,
    hp: Hyperparams,
    writer: SummaryWriter,
    outdir: Path,
    start_round: int,
) -> None:
    timer = Timer(writer, log_first=True)
    gc: Counter = Counter()

    for round in range(start_round, hp.num_rounds):
        logger.info(f"Round {round}")
        policy_model.eval()

        td = get_train_data(
            hp=hp,
            settings=settings,
            device=device,
            policy_model=policy_model,
            timer=timer,
            round=round,
        )

        mean_reward = td["rewards"].mean()
        writer.add_scalar("rewards/reward_mean", mean_reward, round)
        mean_frac_success = td["frac_success"].mean()
        assert mean_frac_success <= 1.0
        writer.add_scalar("rewards/frac_success_mean", mean_frac_success, round)

        writer.add_hparams(
            hparam_dict={
                k: v
                for k, v in asdict(hp).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            metric_dict={
                "metrics/reward_mean": mean_reward,
                "metrics/frac_success": mean_frac_success,
            },
            global_step=round,
        )

        timer.start("advantage")
        compute_advantage(td, hp.gae_gamma, hp.gae_lambda, value_model)
        for k in ["value", "value_target", "unnorm_advantage"]:
            writer.add_scalar(f"rewards/{k}_mean", td[f"{k}s"].mean(), round)
            writer.add_scalar(f"rewards/{k}_std", td[f"{k}s"].std(), round)
        timer.finish("advantage", round)

        timer.start("training")
        train_one_round(
            round=round,
            policy_model=policy_model,
            value_model=value_model,
            td=td,
            hp=hp,
            policy_optim=policy_optim,
            value_optim=value_optim,
            gc=gc,
            writer=writer,
            log_epoch=(hp.num_rounds <= 5),
            outdir=outdir,
        )
        timer.finish("training", round)

        torch.save(
            {
                "round": round + 1,
                "td": td,
                "policy_model": policy_model.state_dict(),
                "value_model": value_model.state_dict(),
                "policy_optim": policy_optim.state_dict(),
                "value_optim": value_optim.state_dict(),
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
    "--autoindex-runs",
    is_flag=True,
    help="Auto-index new runs.",
)
@click.option(
    "--copy-dir",
    type=Path,
    help="Copy contents of a different outdir",
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
) -> None:
    outdir = outdir.resolve()
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
        models = get_models(hp, settings)
        for m in models.values():
            m.to(device)
        policy_model = cast(PolicyModel, models["policy"])
        value_model = cast(ValueModel, models["value"])
        policy_optim = create_optim([policy_model], hp.lr, hp.weight_decay)
        value_optim = create_optim([value_model], hp.lr, hp.weight_decay)
        for obj, key in [
            (policy_model, "policy_model"),
            (value_model, "value_model"),
            (policy_optim, "policy_optim"),
            (value_optim, "value_optim"),
        ]:
            if key in orig_state_dict:
                obj.load_state_dict(orig_state_dict[key])
                del orig_state_dict[key]
        logger.info(
            f"Num Parameters: policy={num_params(policy_model):.2e} value={num_params(value_model):.2e}"
        )

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
                policy_model,
                value_model,
                policy_optim,
                value_optim,
                settings,
                hp,
                writer,
                outdir,
                orig_state_dict.get("round", 0),
            )
    finally:
        (outdir / "_running").unlink()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
