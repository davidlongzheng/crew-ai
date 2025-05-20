from __future__ import absolute_import

import random

import pytest

import cpp_game

from ..engine import Engine
from ..settings import Settings
from ..types import Action, Card


def test_engine_initialization() -> None:
    engine = Engine(
        settings=Settings(
            num_players=3,
            num_side_suits=3,
            side_suit_length=6,
            trump_suit_length=3,
            use_trump_suit=True,
            use_signals=False,
            task_idxs=(0,),
        ),
        seed=42,
    )

    # Check initial state
    assert engine.state.phase == "play"
    assert len(engine.state.hands) == 3
    assert len(engine.state.actions) == 0
    assert engine.state.trick == 0
    assert 0 <= engine.state.leader < 3
    assert engine.state.cur_player == engine.state.leader


def test_hand_generation() -> None:
    engine = Engine(
        settings=Settings(
            num_players=4,
            num_side_suits=2,
            side_suit_length=3,
            trump_suit_length=2,
            use_trump_suit=True,
            use_signals=False,
            task_idxs=(0,),
        ),
        seed=42,
    )

    rng = cpp_game.Rng()
    hands = engine.gen_hands(rng)

    # Check basic hand properties
    assert len(hands) == 4
    total_cards = 0
    for hand in hands:
        assert all(isinstance(card, Card) for card in hand)
        total_cards += len(hand)

    # Total cards should be (num_side_suits * side_suit_length + trump_suit_length)
    expected_total = (2 * 3) + 2  # 8 cards total
    assert total_cards == expected_total


def test_valid_actions() -> None:
    engine = Engine(
        settings=Settings(
            num_players=2,
            num_side_suits=1,
            side_suit_length=2,
            trump_suit_length=1,
            use_trump_suit=True,
            use_signals=False,
            task_idxs=(0,),
        ),
        seed=42,
    )

    # Get valid actions for current player
    actions = engine.valid_actions()
    assert isinstance(actions, list)
    assert all(isinstance(action, Action) for action in actions)
    assert all(action.player == engine.state.cur_player for action in actions)
    assert all(action.type == "play" for action in actions)


def test_play_card() -> None:
    engine = Engine(
        settings=Settings(
            num_players=2,
            num_side_suits=1,
            side_suit_length=2,
            trump_suit_length=1,
            use_trump_suit=True,
            use_signals=False,
            task_idxs=(0,),
        ),
        seed=42,
    )

    initial_player = engine.state.cur_player
    valid_actions = engine.valid_actions()
    assert len(valid_actions) > 0

    # Play first valid action
    action = valid_actions[0]
    engine.move(action)

    assert action.card not in engine.state.hands[initial_player]
    assert len(engine.state.actions) == 1
    assert engine.state.actions[0] == action


def test_follow_suit() -> None:
    # Create a controlled test scenario
    engine = Engine(
        settings=Settings(
            num_players=2,
            num_side_suits=2,
            side_suit_length=2,
            trump_suit_length=1,
            use_trump_suit=True,
            use_signals=False,
            task_idxs=(0,),
        ),
        seed=42,
    )

    # Force specific hands for testing
    engine.state.hands = [
        [Card(rank=1, suit=0), Card(rank=2, suit=1)],
        [Card(rank=2, suit=0), Card(rank=1, suit=1)],
    ]
    engine.state.leader = 0
    engine.state.cur_player = 0

    # Play first card (suit 0)
    first_action = Action(player=0, type="play", card=engine.state.hands[0][0])
    engine.move(first_action)

    # Second player must follow suit
    valid_actions = engine.valid_actions()
    assert len(valid_actions) == 1
    assert valid_actions[0].card is not None
    assert valid_actions[0].card.suit == 0  # Must play the card of suit 0


@pytest.mark.parametrize("use_signals", [True, False])
def test_full_hand(use_signals: bool) -> None:
    engine = Engine(
        settings=Settings(
            use_signals=use_signals,
            task_idxs=(0,),
        ),
        seed=42,
    )
    rng = random.Random(42)

    for trick in range(engine.settings.num_tricks):
        assert trick == engine.state.trick

        # Handle signaling phase if enabled
        while engine.state.phase == "signal":
            valid_actions = engine.valid_actions()
            assert len(valid_actions) > 0
            assert all(
                action.type in ["signal", "nosignal"] for action in valid_actions
            )
            action = rng.choice(valid_actions)
            engine.move(action)

            if action.type == "signal":
                signal = engine.state.signals[action.player]
                assert signal is not None
                assert signal.card == action.card
                assert signal.trick == trick

        # Handle play phase
        assert engine.state.phase == "play"
        for turn_idx in range(engine.settings.num_players):
            assert (
                engine.state.cur_player
                == (engine.state.leader + turn_idx) % engine.settings.num_players
            )
            valid_actions = engine.valid_actions()
            assert len(valid_actions) > 0
            assert all(action.type == "play" for action in valid_actions)

            action = rng.choice(valid_actions)
            engine.move(action)

            assert action.card not in engine.state.hands[action.player]
            assert engine.state.actions[-1] == action

            if turn_idx != engine.settings.num_players - 1:
                assert len(
                    engine.state.active_cards
                ) == turn_idx + 1 and engine.state.active_cards[-1] == (
                    action.card,
                    action.player,
                )
            else:
                assert len(engine.state.active_cards) == 0

    assert engine.state.hands == [[] for _ in range(engine.settings.num_players)]
    assert engine.state.phase == "end"


def test_draft() -> None:
    engine = Engine(
        settings=Settings(
            use_drafting=True,
            bank="med",
            min_difficulty=4,
            max_difficulty=7,
            max_num_tasks=4,
        ),
        seed=44,
    )
    print(engine.state.task_idxs)

    while engine.state.phase != "end":
        valid_actions = engine.valid_actions()
        action = valid_actions[-1]
        engine.move(action)

    print(engine.state.actions)
