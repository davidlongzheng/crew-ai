import re
import shutil
import subprocess
import sys
from functools import cache, partial
from pathlib import Path

import click
import numpy as np
import optuna
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler  # bayesian sampler
from optuna.trial import TrialState

EVAL_REGEX = re.compile(r"Round (\d+):.*win_rate=([-\d\.]+)")
MODEL_SIZE_REGEX = re.compile(r"Num Parameters: pv=([\d\.e\+]+)")

HPARAMS = {
    "embed_dim": dict(min=32, max=128, n=8, step=16, log=True),
    "tasks_hidden_dim": dict(min="embed_dim", max=128, n=8, step=16, log=True),
    "tasks_embed_dim": dict(min="embed_dim", max="tasks_hidden_dim", n=8, step=16),
    "hand_hidden_dim": dict(min="embed_dim", max=128, n=8, step=16, log=True),
    "hand_embed_dim": dict(min="embed_dim", max="hand_hidden_dim", n=8, step=16),
    "hist_hidden_dim": dict(min="embed_dim", max=256, n=8, step=16, log=True),
    "hist_output_dim": dict(min="embed_dim", max="hist_hidden_dim", n=8, step=16),
    "backbone_hidden_dim": dict(min=256, max=1024, n=8, step=64, log=True),
    "backbone_output_dim": dict(min=8, max=64, n=8, step=8),
    "policy_hidden_dim": dict(min="embed_dim", max=256, n=8, step=16, log=True),
    "policy_query_dim": dict(min=8, max="policy_hidden_dim", n=8, step=8),
    "lr": dict(min=1e-4, max=2e-3, n=8, step=1e-4, log=True),
    "embed_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "hand_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "tasks_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "hist_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "backbone_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "policy_dropout": dict(min=0.0, max=0.06, n=8, step=0.01),
    "weight_decay": dict(min=1e-4, max=1e-2, n=8, step=1e-3, log=True),
    "gae_lambda": dict(min=0.9, max=0.99, n=8, step=0.01),
    "grad_norm_clip": dict(min=0.1, max=10, n=8, step=0.1, log=True),
    "hist_num_layers": dict(choices=[1, 2]),
    "aux_info_coef": dict(choices=[0, 0.01, 0.1]),
}


@cache
def get_choices(name):
    params = HPARAMS[name]
    if "choices" in params:
        return params["choices"]

    min = (
        HPARAMS[params["min"]]["min"]
        if isinstance(params["min"], str)
        else params["min"]
    )
    max = (
        HPARAMS[params["max"]]["max"]
        if isinstance(params["max"], str)
        else params["max"]
    )
    n = params["n"]
    step = params["step"]
    log = params.get("log", False)
    choices = np.geomspace(min, max, n) if log else np.linspace(min, max, n)
    choices = [round(x / step) * step for x in choices]
    if isinstance(step, int):
        choices = [int(x) for x in choices]

    new_choices = []
    for x in choices:
        if new_choices and abs(x - new_choices[-1]) <= 1e-8:
            continue
        new_choices.append(x)

    return new_choices


@cache
def get_dist_func(name):
    choices = get_choices(name)

    def dist_func(val1, val2):
        new_val1 = min(choices, key=lambda x: abs(val1 - x))
        new_val2 = min(choices, key=lambda x: abs(val2 - x))
        assert abs(val1 - new_val1) <= 1e-8
        assert abs(val2 - new_val2) <= 1e-8
        return abs(choices.index(new_val1) - choices.index(new_val2))

    return dist_func


def suggest_categorical(
    trial,
    name,
):
    # params = HPARAMS[name]
    choices = get_choices(name)
    # no support for categorical dynamic value spaces right now.
    # if isinstance(params.get("min"), str):
    #     choices = [x for x in choices if x >= trial.params[params["min"]]]
    # if isinstance(params.get("max"), str):
    #     choices = [x for x in choices if x <= trial.params[params["max"]]]

    trial.suggest_categorical(name, choices)


def objective(trial: optuna.Trial, outdir: Path, num_rounds: int | None) -> float:
    for name in HPARAMS:
        suggest_categorical(trial, name)

    hp_str = ",".join(f"{k}={v}" for k, v in trial.params.items())
    if num_rounds is not None:
        hp_str += f",num_rounds={num_rounds}"
    run_dir = outdir / f"run_{trial.number}"

    cmd = [
        sys.executable,
        "-m",
        "src.ai.train",
        "--outdir",
        str(run_dir),
        "--hp",
        hp_str,
        "--skip-checkpoint",
    ]

    with subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        assert proc.stderr is not None
        best_win_rate = 0.0
        model_size = None
        stderr_lines = []
        for line in proc.stderr:
            stderr_lines.append(line)
            if m := MODEL_SIZE_REGEX.search(line):
                assert model_size is None
                model_size = float(m.group(1))
                trial.set_user_attr("model_size", model_size)
            elif m := EVAL_REGEX.search(line):
                round, win_rate = int(m.group(1)), float(m.group(2))
                best_win_rate = max(best_win_rate, win_rate)
                trial.report(best_win_rate, round)
                if trial.should_prune():
                    proc.terminate()
                    logger.warning(
                        f"Pruned trial {trial.number} b/c best_win_rate {best_win_rate} worse than median at round {round}"
                    )
                    raise optuna.TrialPruned()

                for check_round, cutoff in [
                    (5, 0.2),
                    (10, 0.30),
                    (20, 0.50),
                    (30, 0.55),
                    (40, 0.60),
                    (50, 0.65),
                    (75, 0.70),
                ]:
                    if round >= check_round and best_win_rate < cutoff:
                        logger.warning(
                            f"Pruned trial {trial.number} b/c best_win_rate {best_win_rate} < {cutoff} at round {round}"
                        )
                        raise optuna.TrialPruned()

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Training failed:\nStderr:\n{''.join(stderr_lines)}")

    assert model_size is not None
    return best_win_rate


@click.command()
@click.option(
    "--root-dir",
    type=Path,
    help="Root directory",
    required=True,
)
@click.option(
    "--study-name",
    type=str,
    help="Study name",
    required=True,
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Number of trials to run",
)
@click.option(
    "--n-jobs",
    type=int,
    default=10,
    help="Number of jobs to run in parallel",
)
@click.option(
    "--clean",
    is_flag=True,
    help="Clean outdir",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from existing outdir",
)
@click.option(
    "--num-rounds",
    type=int,
    default=None,
    help="Number of rounds to run",
)
def main(
    root_dir: Path,
    study_name: str,
    n_trials: int,
    n_jobs: int,
    clean: bool,
    resume: bool,
    num_rounds: int | None,
):
    root_dir = root_dir.resolve()
    outdir = root_dir / study_name
    if outdir.exists():
        if clean:
            logger.info(f"** Cleaning outdir {outdir} **")
            shutil.rmtree(outdir)
        elif not resume:
            raise Exception("Must set --clean or --resume to run on existing outdir.")

    outdir.mkdir(parents=True, exist_ok=True)
    logger.add(outdir / "optimize.log")

    logger.info("Creating study")
    categorical_dists = {name: get_dist_func(name) for name in HPARAMS}
    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize"],
        sampler=TPESampler(
            multivariate=True, categorical_distance_func=categorical_dists
        ),
        pruner=MedianPruner(
            n_startup_trials=10,
            n_min_trials=10,
            n_warmup_steps=30,
            interval_steps=10,
        ),
        storage=f"sqlite:///{root_dir}/optuna.db",
        load_if_exists=True,
    )
    # Doesn't fit into search space step unfortunately ...
    # hp = Hyperparams()
    # study.enqueue_trial({k: getattr(hp, k) for k in HPARAMS})
    logger.info("Optimizing")
    study.optimize(
        partial(objective, outdir=outdir, num_rounds=num_rounds),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    logger.info("Best trial:")
    best_trial = study.best_trial

    logger.info(f"  Value: {best_trial.value}")

    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    logger.info(f"  Model size: {best_trial.user_attrs['model_size']}")


if __name__ == "__main__":
    main()
