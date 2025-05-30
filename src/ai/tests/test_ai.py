import copy
from dataclasses import replace

from ai.ai import get_ai
from game.engine import Engine
from game.settings import get_preset


def test_ai():
    settings = replace(
        get_preset("med"),
        min_difficulty=7,
        max_difficulty=7,
    )

    ai = get_ai(settings)
    ai_state = ai.new_rollout()

    engine = Engine(settings, seed=42)
    while engine.state.phase != "end":
        action = ai.get_move(engine, ai_state)
        engine.move(action)


def test_ai_tree_search():
    settings = replace(
        get_preset("med"),
        min_difficulty=7,
        max_difficulty=7,
    )

    ai = get_ai(settings)
    ai_state = ai.new_rollout()

    engine = Engine(settings, seed=42)

    while engine.state.phase == "draft":
        action = ai.get_move(engine, ai_state)
        engine.move(action)

    while engine.state.phase != "end":
        ai_state_copy = copy.deepcopy(ai_state)
        old_action = ai.get_move(engine, ai_state_copy)
        action = ai.get_move_tree_search(engine, ai_state)
        if old_action != action:
            print("Old:", old_action, "New:", action)
        engine.move(action)

    print(engine.state.status)
