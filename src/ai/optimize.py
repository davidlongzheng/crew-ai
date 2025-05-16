import re
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import click
import optuna
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler  # bayesian sampler
from optuna.trial import TrialState

from .hyperparams import Hyperparams

EVAL_REGEX = re.compile(r"Round (\d+):.*win_rate=([-\d\.]+)")
MODEL_SIZE_REGEX = re.compile(r"Num Parameters: pv=([\d\.e\+]+)")


def objective(trial: optuna.Trial, outdir: Path, num_rounds: int | None) -> float:
    # ----- 1. sample hyper-parameters -----------------------
    embed_dim = trial.suggest_int("embed_dim", 16, 128, step=16)
    tasks_hidden_dim = trial.suggest_int("tasks_hidden_dim", embed_dim, 128, step=16)
    trial.suggest_int("tasks_embed_dim", embed_dim, tasks_hidden_dim, step=16)
    hand_hidden_dim = trial.suggest_int("hand_hidden_dim", embed_dim, 128, step=16)
    trial.suggest_int("hand_embed_dim", embed_dim, hand_hidden_dim, step=16)
    hist_hidden_dim = trial.suggest_int("hist_hidden_dim", embed_dim, 256, step=16)
    trial.suggest_int("hist_output_dim", embed_dim, hist_hidden_dim, step=16)
    backbone_hidden_dim = trial.suggest_int(
        "backbone_hidden_dim", embed_dim, 512, step=16
    )
    trial.suggest_int("backbone_output_dim", 8, backbone_hidden_dim, step=8)
    policy_hidden_dim = trial.suggest_int("policy_hidden_dim", embed_dim, 128, step=16)
    trial.suggest_int("policy_query_dim", 8, policy_hidden_dim, step=8)

    trial.suggest_int("batch_size", 32, 256, step=32)
    trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    trial.suggest_float("dropout", 0.0, 0.1)
    trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    trial.suggest_float("gae_lambda", 0.85, 0.99)
    trial.suggest_float("grad_norm_clip", 0.1, 10, log=True)

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

                # if model_size <= 1e4 or model_size >= 1e6:
                #     logger.warning(
                #         f"Pruned trial {trial.number} b/c model_size={model_size:.2e}"
                #     )
                #     raise optuna.TrialPruned()
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

                # for check_round, cutoff in [
                #     (5, 0.2),
                #     (10, 0.30),
                #     (20, 0.50),
                #     (30, 0.55),
                #     (40, 0.60),
                #     (50, 0.65),
                #     (75, 0.70),
                # ]:
                #     if round >= check_round and best_win_rate < cutoff:
                #         logger.warning(
                #             f"Pruned trial {trial.number} b/c best_win_rate {best_win_rate} < {cutoff} at round {round}"
                #         )
                #         raise optuna.TrialPruned()

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
    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize"],
        sampler=TPESampler(multivariate=True),  # captures interactions
        pruner=MedianPruner(
            n_startup_trials=10,
            n_min_trials=10,
            n_warmup_steps=50,
            interval_steps=10,
        ),
        storage=f"sqlite:///{root_dir}/optuna.db",
        load_if_exists=True,
    )
    hp = Hyperparams()
    study.enqueue_trial(
        {
            k: getattr(hp, k)
            for k in [
                "embed_dim",
                "tasks_hidden_dim",
                "tasks_embed_dim",
                "hand_hidden_dim",
                "hand_embed_dim",
                "hist_hidden_dim",
                "hist_output_dim",
                "backbone_hidden_dim",
                "backbone_output_dim",
                "policy_hidden_dim",
                "policy_query_dim",
                "batch_size",
                "lr",
                "weight_decay",
                "gae_lambda",
                "grad_norm_clip",
            ]
        }
        | {
            "dropout": hp.hist_dropout,
        },
    )
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
