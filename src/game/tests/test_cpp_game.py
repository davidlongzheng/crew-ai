import random
from dataclasses import replace

import pytest

import cpp_game
from game.engine import Engine
from game.settings import get_preset
from game.tasks import TASK_DEFS


def assert_matching_settings(settings, cpp_settings):
    for field in [
        "num_players",
        "num_side_suits",
        "use_trump_suit",
        "side_suit_length",
        "trump_suit_length",
        "use_signals",
        "single_signal",
        "bank",
        "task_distro",
        "min_difficulty",
        "max_difficulty",
        "max_num_tasks",
        "task_bonus",
        "win_bonus",
        "use_drafting",
        "num_draft_tricks",
    ]:
        assert getattr(settings, field) == getattr(cpp_settings, field)

    assert list(settings.task_idxs) == cpp_settings.task_idxs


def assert_matching_tasks(engine, cpp_engine):
    tasks = engine.state.assigned_tasks
    cpp_tasks = cpp_engine.state.assigned_tasks

    assert len(tasks) == len(cpp_tasks)
    for player_tasks, cpp_player_tasks in zip(tasks, cpp_tasks):
        assert len(player_tasks) == len(cpp_player_tasks)
        for task, cpp_task in zip(player_tasks, cpp_player_tasks):
            assert task.formula == cpp_task.formula, (task.formula, cpp_task.formula)
            assert task.desc == cpp_task.desc
            assert task.difficulty == cpp_task.difficulty
            assert task.task_idx == cpp_task.task_idx
            assert task.player == cpp_task.player
            assert to_cpp_status(task.status) == cpp_task.status
            assert task.value == cpp_task.value
            assert task.in_one_trick == cpp_task.in_one_trick


def init_matching_engines(
    seed, task_idxs, single_signal, use_drafting, weight_by_difficulty
):
    settings = get_preset("easy")
    settings = replace(
        settings,
        bank="all",
        task_idxs=tuple(task_idxs),
        use_signals=True,
        single_signal=single_signal,
        use_drafting=use_drafting,
        weight_by_difficulty=weight_by_difficulty,
    )
    cpp_settings = settings.to_cpp()
    assert_matching_settings(settings, cpp_settings)
    assert settings.task_idxs
    assert cpp_settings.task_idxs

    engine = Engine(
        settings=settings,
        seed=seed,
    )
    cpp_engine = cpp_game.Engine(
        settings=cpp_settings,
        seed=seed,
    )
    assert_matching_tasks(engine, cpp_engine)
    assert_matching_state(engine, cpp_engine)

    return engine, cpp_engine


def to_cpp_card(card):
    return cpp_game.Card(card.rank, card.suit)


def to_cpp_action(action):
    return cpp_game.Action(
        action.player,
        to_cpp_action_type(action.type),
        to_cpp_card(action.card) if action.card else None,
        action.task_idx,
    )


def to_cpp_signal(signal):
    return cpp_game.Signal(
        to_cpp_card(signal.card), to_cpp_sig_value(signal.value), signal.trick
    )


def to_cpp_action_type(action_type):
    return getattr(cpp_game.ActionType, action_type)


def to_cpp_sig_value(sig_value):
    return getattr(cpp_game.SignalValue, sig_value)


def to_cpp_phase(phase):
    return getattr(cpp_game.Phase, phase)


def to_cpp_status(status):
    return getattr(cpp_game.Status, status)


def to_cpp_last_action(last_action):
    if last_action is None:
        return None

    last_action = list(last_action)
    last_action[2] = to_cpp_action(last_action[2])
    last_action = tuple(last_action)
    return last_action


def assert_matching_state(engine, cpp_engine):
    state = engine.state
    cpp_state = cpp_engine.state

    assert to_cpp_phase(state.phase), cpp_state.phase
    assert [[to_cpp_card(x) for x in hand] for hand in state.hands] == cpp_state.hands
    assert to_cpp_last_action(state.last_action) == cpp_state.last_action
    assert state.trick == cpp_state.trick
    assert state.leader == cpp_state.leader
    assert state.captain == cpp_state.captain
    assert state.cur_player == cpp_state.cur_player
    assert [
        (to_cpp_card(x), y) for x, y in state.active_cards
    ] == cpp_state.active_cards
    assert [
        ([to_cpp_card(c) for c in x], y) for x, y in state.past_tricks
    ] == cpp_state.past_tricks
    assert [to_cpp_signal(x) if x else None for x in state.signals] == cpp_state.signals
    assert state.trick_winner == cpp_state.trick_winner
    assert to_cpp_status(state.status) == cpp_state.status, (
        state.status,
        cpp_state.status,
    )
    assert abs(state.value - cpp_state.value) <= 1e-8
    assert_matching_tasks(engine, cpp_engine)


@pytest.mark.parametrize("single_signal", [True, False])
@pytest.mark.parametrize("use_drafting", [True, False])
@pytest.mark.parametrize("weight_by_difficulty", [True, False])
def test_cpp_game(single_signal, use_drafting, weight_by_difficulty):
    num_tasks = len(TASK_DEFS)
    step_size = 8
    for task_start_idx in range(0, num_tasks, step_size):
        task_idxs = list(
            range(task_start_idx, min(task_start_idx + step_size, num_tasks))
        )
        for engine_seed in range(5):
            engine, cpp_engine = init_matching_engines(
                seed=engine_seed,
                task_idxs=task_idxs,
                single_signal=single_signal,
                use_drafting=use_drafting,
                weight_by_difficulty=weight_by_difficulty,
            )

            assert_matching_state(engine, cpp_engine)

            rng = random.Random(42)

            while engine.state.phase != "end":
                assert cpp_engine.state.phase != "end"

                valid_actions = engine.valid_actions()
                assert len(valid_actions) > 0

                cpp_valid_actions = cpp_engine.valid_actions()
                assert len(cpp_valid_actions) > 0

                assert len(valid_actions) == len(cpp_valid_actions)
                assert all(
                    to_cpp_action(x) == y
                    for x, y in zip(valid_actions, cpp_valid_actions)
                )

                action = rng.choice(valid_actions)
                cpp_action = to_cpp_action(action)
                engine.move(action)
                cpp_engine.move(cpp_action)

                assert_matching_state(engine, cpp_engine)
