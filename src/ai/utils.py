import time
from collections import defaultdict
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ai.hyperparams import Hyperparams
from game.settings import Settings
from game.utils import get_splits_and_phases


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None,
        num_hidden_layers: int,
        *,
        dropout=0.0,
        use_layer_norm=False,
        activation=torch.nn.GELU,
        use_resid=False,
    ):
        super().__init__()
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout else None

        fcs = []
        layer_norms = []
        resid_fcs = []

        assert num_hidden_layers >= 1

        def make_layer(inp_dim, out_dim):
            fcs.append(nn.Linear(inp_dim, out_dim))
            if use_layer_norm:
                layer_norms.append(nn.LayerNorm(out_dim))
            if use_resid:
                if inp_dim == out_dim:
                    resid_fcs.append(nn.Identity())
                else:
                    resid_fcs.append(nn.Linear(inp_dim, out_dim))

        make_layer(input_dim, hidden_dim)
        for _ in range(num_hidden_layers - 1):
            make_layer(hidden_dim, hidden_dim)

        self.fcs = nn.ModuleList(fcs)
        self.layer_norms = nn.ModuleList(layer_norms) if use_layer_norm else None
        self.resid_fcs = nn.ModuleList(resid_fcs) if use_resid else None
        self.out_fc = (
            nn.Linear(hidden_dim, output_dim) if output_dim is not None else None
        )

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            y = self.activation(fc(x))
            if self.dropout:
                y = self.dropout(y)
            if self.layer_norms:
                y = self.layer_norms[i](y)
            if self.resid_fcs:
                y = self.resid_fcs[i](x) + y
            x = y

        if self.out_fc:
            x = self.out_fc(x)

        return x


def win_rate_by_difficulty(td):
    win = td["win"].float()
    difficulty = td["difficulty"]
    unique_cats, bin_indices = torch.unique(difficulty, return_inverse=True)

    bin_sums = torch.bincount(bin_indices, weights=win)
    bin_counts = torch.bincount(bin_indices)
    bin_means = bin_sums / bin_counts

    return dict(zip(unique_cats.tolist(), bin_means.tolist()))


def set_lstm_state(pv_model, lstm_states):
    if all(x is None for x in lstm_states):
        pv_model.set_state(None)
        return

    assert all(x is not None for x in lstm_states)
    lstm_state = tuple([torch.cat(x, dim=1) for x in zip(*lstm_states)])
    pv_model.set_state(lstm_state)


def get_lstm_state(pv_model):
    lstm_state = pv_model.get_state()
    for i in range(lstm_state[0].size(1)):
        yield tuple([x[:, i : i + 1] for x in lstm_state])


def print_memory_usage(key):
    device = torch.device(0)
    alloc = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    logger.info(f"Memory Usage ({key}): alloc={alloc:.2f}MB reserved={reserved:.2f}MB")


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    return torch.device(0 if torch.cuda.is_available() else "cpu")


def should_keep_outdir(x):
    return (x / "_running").exists() or (x / "_keep").exists()


def create_optim(
    models, hp: Hyperparams, private: bool = False
) -> torch.optim.Optimizer:
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
        lr=(hp.private_lr if private else hp.lr),
        weight_decay=hp.weight_decay,
        betas=(hp.beta_1, hp.beta_2),
    )
    return optimizer


def create_lr_sched(
    optimizer: torch.optim.Optimizer,
    hp: Hyperparams,
    private: bool = False,
) -> torch.optim.lr_scheduler.LRScheduler:
    if hp.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp.num_rounds,
            eta_min=(hp.private_lr if private else hp.lr) * hp.lr_min_frac,
        )
    elif hp.lr_schedule == "warmup_linear":
        num_warmup_rounds = int(hp.num_rounds * hp.lr_warmup_frac)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: x / num_warmup_rounds
            if x < num_warmup_rounds
            else 1
            - (x - num_warmup_rounds)
            / (hp.num_rounds - num_warmup_rounds - 1)
            * (1 - hp.lr_min_frac),
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


def get_phase_weights(settings: Settings, hp: Hyperparams, device: torch.device):
    splits, phases = get_splits_and_phases(settings, as_phase_index=False)
    phases_t = [[phase] * split for split, phase in zip(splits, phases)]
    phases_t = [x for y in phases_t for x in y]
    assert len(phases_t) == settings.get_seq_length()
    weight_dict = {
        "play": 1.0,
        "signal": hp.signal_weight,
        "draft": hp.draft_weight,
    }
    weights_t = [weight_dict[x] for x in phases_t]
    if all(x == 1.0 for x in weights_t):
        return None

    weights = torch.tensor(weights_t, device=device)
    weights /= weights.mean()

    return weights
