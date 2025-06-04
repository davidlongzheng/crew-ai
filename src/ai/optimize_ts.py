import shutil
from dataclasses import replace
from pathlib import Path

import click
import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler  # bayesian sampler
from optuna.trial import TrialState

import cpp_game
from ai.ai import AI
from ai.tree_search import TreeSearchSettings
from game.settings import DEFAULT_PRESET, get_preset

NUM_ROLLOUTS = 200


def tree_rollout(engine, ai, seed=None):
    engine.reset_state(seed)
    ai_state = ai.new_rollout()

    while engine.state.phase == cpp_game.Phase.draft:
        action = ai.get_move(engine, ai_state)
        engine.move(action)

    while engine.state.phase != cpp_game.Phase.end:
        action = ai.get_move_tree_search(engine, ai_state)
        engine.move(action)

    return engine.state.status == cpp_game.Status.success


def objective(trial: optuna.Trial) -> float:
    c_puct_init = trial.suggest_float("c_puct_init", 1.0, 2.0, step=0.05)
    num_parallel = trial.suggest_int("num_parallel", 5, 20, step=1)
    root_noise = trial.suggest_categorical("root_noise", [True, False])
    if root_noise:
        all_noise = trial.suggest_categorical("all_noise", [True, False])
    else:
        all_noise = False

    if root_noise:
        noise_eps = trial.suggest_float("noise_eps", 0.1, 0.4, step=0.05)
        noise_alpha = trial.suggest_float("noise_alpha", 0.01, 0.5, log=True)
    else:
        noise_eps = 0.0
        noise_alpha = 0.0

    use_skip_thresh = trial.suggest_categorical("use_skip_thresh", [True, False])

    settings = replace(
        get_preset(DEFAULT_PRESET),
        min_difficulty=7,
        max_difficulty=7,
    )

    for path in [
        Path("/Users/davidzheng/projects/crew-ai/outdirs/0525/run_7"),
        Path("/root/outdirs/0531/run_2"),
    ]:
        if path.exists():
            ai = AI(path)
            break
    else:
        raise ValueError("Paths do not exist")

    cpp_settings = settings.to_cpp()
    engine = cpp_game.Engine(cpp_settings)

    ts_settings = TreeSearchSettings(
        c_puct_base=19652,
        c_puct_init=c_puct_init,
        num_parallel=num_parallel,
        root_noise=root_noise,
        all_noise=all_noise,
        cheating=True,
        noise_eps=noise_eps,
        noise_alpha=noise_alpha,
        skip_thresh=0.98 if use_skip_thresh else None,
        num_iters=200,
        seed=42,
    )
    assert ai.set_ts_settings(ts_settings)
    assert ts_settings == ai.ts_settings
    logger.info(f"ts_settings: {ai.ts_settings}")

    wins = []
    for i in range(NUM_ROLLOUTS):
        assert ts_settings == ai.ts_settings
        seed = i + 42
        wins.append(tree_rollout(engine, ai, seed))
        if i < 5:
            continue
        win_rate = np.mean(wins)
        trial.report(win_rate, i)

    return win_rate


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
    default=1,
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
def main(
    root_dir: Path,
    study_name: str,
    n_trials: int,
    n_jobs: int,
    clean: bool,
    resume: bool,
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
        sampler=TPESampler(),
        storage=f"sqlite:///{root_dir}/optuna.db",
        load_if_exists=True,
    )
    logger.info("Optimizing")
    study.optimize(
        objective,
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


if __name__ == "__main__":
    main()
