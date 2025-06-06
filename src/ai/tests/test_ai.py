import copy
import time
from dataclasses import replace

import numpy as np

import cpp_game
from ai.ai import batch_rollout, get_ai
from ai.tree_search import TreeSearchSettings
from game.settings import DEFAULT_PRESET, get_preset


def test_ai():
    settings = replace(
        get_preset(DEFAULT_PRESET),
        min_difficulty=4,
        max_difficulty=4,
    )

    ai = get_ai(settings)
    ai_state = ai.new_rollout()

    cpp_settings = settings.to_cpp()
    engine = cpp_game.Engine(cpp_settings, seed=42)
    while engine.state.phase != cpp_game.Phase.end:
        action = ai.get_move(engine, ai_state)
        engine.move(action)

    assert engine.state.status == cpp_game.Status.success


def test_ai_tree_search():
    settings = replace(
        get_preset(DEFAULT_PRESET),
        min_difficulty=7,
        max_difficulty=7,
    )

    ai = get_ai(settings)
    ai_state = ai.new_rollout()

    cpp_settings = settings.to_cpp()
    engine = cpp_game.Engine(cpp_settings, seed=42)

    while engine.state.phase == cpp_game.Phase.draft:
        action = ai.get_move(engine, ai_state)
        engine.move(action)

    while engine.state.phase != cpp_game.Phase.end:
        ai_state_copy = copy.deepcopy(ai_state)
        old_action = ai.get_move(engine, ai_state_copy)
        action = ai.get_move_tree_search(engine, ai_state)
        if old_action != action:
            print("Old:", old_action, "New:", action)
        engine.move(action)

    print(engine.state.status)


def test_ai_batch_rollout():
    num_rollouts = 10
    settings = replace(
        get_preset(DEFAULT_PRESET),
        min_difficulty=7,
        max_difficulty=7,
    )
    cpp_settings = settings.to_cpp()
    ts_settings = TreeSearchSettings(
        num_iters=100,
        seed=42,
    )
    ai = get_ai(settings, ts_settings, num_rollouts)

    wins = []
    seeds = list(range(42, 42 + num_rollouts))
    engines = [cpp_game.Engine(cpp_settings) for _ in range(num_rollouts)]

    wins = batch_rollout(engines, ai, seeds)

    print(
        f"n={len(wins)}, win={np.mean(wins):.3f}±{np.std(wins) / np.sqrt(len(wins)):.3f}"
    )


def test_ai_tree_search_batch_rollout():
    num_rollouts = 10
    settings = replace(
        get_preset(DEFAULT_PRESET),
        min_difficulty=7,
        max_difficulty=7,
    )
    cpp_settings = settings.to_cpp()
    ts_settings = TreeSearchSettings(
        num_iters=100,
        seed=42,
    )
    ai = get_ai(settings, ts_settings, num_rollouts)

    wins = []
    tree_wins = []
    seeds = list(range(42, 42 + num_rollouts))
    engines = [cpp_game.Engine(cpp_settings) for _ in range(num_rollouts)]

    wins = batch_rollout(engines, ai, seeds)
    start_time = time.time()
    tree_wins = batch_rollout(engines, ai, seeds, use_tree_search=True)
    elapsed = time.time() - start_time

    print(
        f"n={len(wins)}, win={np.mean(wins):.3f}±{np.std(wins) / np.sqrt(len(wins)):.3f} tree_win={np.mean(tree_wins):.3f}±{np.std(tree_wins) / np.sqrt(len(tree_wins)):.3f} elapsed={elapsed:.3f}"
    )
