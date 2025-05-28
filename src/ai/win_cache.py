"""Cache the win rate of each engine seed."""

import multiprocessing
import random
from functools import partial
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from ai.rollout import do_batch_rollout
from game.settings import DEFAULT_PRESET, SETTINGS_TYPE, Settings, get_preset

CANON_WIN_CACHE_DIR = Path(__file__).parent.parent.parent / "win_cache"


def get_cache_list(cache_dir: Path):
    cache_list_fn = cache_dir / "cache_list.pkl"
    if not cache_list_fn.exists():
        cache_list = []
    else:
        cache_list = pd.read_pickle(cache_list_fn)

    return cache_list


def get_cache_idx(cache_list, settings: Settings):
    for idx, _settings in enumerate(cache_list):
        if settings == _settings:
            return idx

    return len(cache_list)


def get_cache_fn(cache_dir, cache_idx):
    return cache_dir / f"cache_{cache_idx}.pkl"


def load_cache(settings: Settings, cache_dir: Path = CANON_WIN_CACHE_DIR):
    cache_list = get_cache_list(cache_dir)
    cache_idx = get_cache_idx(cache_list, settings)
    cache_fn = get_cache_fn(cache_dir, cache_idx)
    if not cache_fn.exists():
        raise Exception(
            f"Could not find cache fn for given settings in {cache_dir}. Use win_cache.py to generate."
        )
    return pd.read_pickle(cache_fn)


def run_one_seed(engine_seed, settings, num_rollouts):
    batch_seed = random.Random(engine_seed).randint(0, 100_000_000)
    rollouts = do_batch_rollout(
        settings=settings,
        num_rollouts=num_rollouts,
        batch_seed=batch_seed,
        engine_seeds=engine_seed,
    )
    return {
        "seed": batch_seed,
        "win_rate": float(np.mean([x["win"] for x in rollouts])),
        "frac_success": float(
            np.mean(
                [
                    sum(y[0] for y in x["num_success_tasks_pp"])
                    / sum(y[1] for y in x["num_success_tasks_pp"])
                    for x in rollouts
                ]
            )
        ),
        "avg_reward": float(np.mean([sum(x["rewards"]) for x in rollouts])),
        "example": rollouts[0],
    }


def run(settings, num_seeds, num_rollouts_per_seed, num_workers):
    rng = random.Random(42)
    engine_seeds = set()
    while len(engine_seeds) < num_seeds:
        engine_seeds.add(rng.randint(0, 100_000_000))
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        run_one_seed,
                        settings=settings,
                        num_rollouts=num_rollouts_per_seed,
                    ),
                    engine_seeds,
                ),
                total=num_seeds,
                desc="Running seeds",
            )
        )

    results = pd.DataFrame(results)
    return results


@click.command()
@click.option(
    "--cache-dir",
    type=Path,
    default=CANON_WIN_CACHE_DIR,
    help="Cache dir",
)
@click.option(
    "--settings",
    type=SETTINGS_TYPE,
    default=get_preset(DEFAULT_PRESET),
    help="Settings",
)
@click.option(
    "--policy-fn",
    type=Path,
    default=None,
)
@click.option(
    "--num-seeds",
    type=int,
    default=1_000_000,
)
@click.option(
    "--num-rollouts-per-seed",
    type=int,
    default=20,
)
@click.option(
    "--num-workers",
    type=int,
    default=16,
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing cache if exists.")
def main(
    cache_dir: Path,
    settings: Settings,
    policy_fn: Path | None,
    num_seeds: int,
    num_rollouts_per_seed: int,
    num_workers: int,
    overwrite: bool,
):
    assert policy_fn is None, "Custom policy not implemented"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_list = get_cache_list(cache_dir)
    cache_idx = get_cache_idx(cache_list, settings)
    if cache_idx >= len(cache_list):
        assert cache_idx == len(cache_list)
        cache_list.append(settings)
    cache_fn = get_cache_fn(cache_dir, cache_idx)
    if cache_fn.exists() and not overwrite:
        raise Exception(f"Must set --overwrite to overwrite {cache_fn}")

    info = run(settings, num_seeds, num_rollouts_per_seed, num_workers)
    pd.to_pickle(info, cache_fn)
    pd.to_pickle(cache_list, cache_dir / "cache_list.pkl")


if __name__ == "__main__":
    main()
