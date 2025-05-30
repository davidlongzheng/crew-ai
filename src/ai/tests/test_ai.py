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

    engine = Engine(settings)
    while engine.state.phase != "end":
        action = ai.get_move(engine, ai_state)
        engine.move(action)
