from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ..settings import Settings
from ..state import State
from ..tasks import TASK_DEFS, AssignedTask
from ..types import Card


@dataclass
class MockState:
    trick: int = 0
    trick_winner: int | None = None
    leader: int = 0
    captain: int = 0
    active_cards: list[tuple[Card, int]] = field(default_factory=list)

    def as_state(self) -> State:
        return cast(State, self)


def test_parse_formula():
    settings = Settings(task_idxs=(0,))
    for formula, desc, difficulty in TASK_DEFS:
        AssignedTask(
            formula=formula,
            desc=desc,
            difficulty=difficulty,
            task_idx=0,
            player=0,
            settings=settings,
        )


def test_sum_condition():
    """Test '1T sum>28 #t=0' - must win a trick with sum > 28 and no trumps"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="1T sum>28 #t=0",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Test failing case - sum too low
    state.active_cards = [
        (Card(9, 0), 0),
        (Card(9, 1), 1),
        (Card(9, 2), 2),
        (Card(1, 3), 3),
    ]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"  # One-trick tasks reset after each trick

    # Test failing case - has trump
    state.active_cards = [
        (Card(9, 0), 0),
        (Card(9, 1), 1),
        (Card(9, 2), 2),
        (Card(4, 4), 3),
    ]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Test successful case
    state.active_cards = [
        (Card(9, 0), 0),
        (Card(9, 1), 1),
        (Card(9, 2), 2),
        (Card(9, 3), 3),
    ]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "success"

    # Test game end without success
    task = AssignedTask(
        formula="1T sum>28 #t=0",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    task.on_game_end()
    assert task.status == "fail"


def test_consecutive_tricks():
    """Test 'consec(3)' - must win 3 consecutive tricks"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="consec(3)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Win first two tricks
    state.trick = 0
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    state.trick = 1
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Lose third trick - breaks consecutiveness
    state.trick = 2
    state.trick_winner = 1
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Win next three tricks consecutively
    state.trick = 3
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    state.trick = 4
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    state.trick = 5
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "success"

    task.on_game_end()
    assert task.status == "success"


def test_specific_tricks():
    """Test 'T0 T1' - must win tricks 0 and 1"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="T0 T1",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Win trick 0
    state.trick = 0
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Win trick 1
    state.trick = 1
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "success"

    # Test failing case
    task = AssignedTask(
        formula="T0 T1",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state.trick = 0
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    state.trick = 1
    state.trick_winner = 1  # Different player wins trick 1
    task.on_trick_end(state.as_state())
    assert task.status == "fail"


def test_specific_card():
    """Test '9p' - must win the 9 of pink"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="9p",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Test not winning trick with card
    state.active_cards = [(Card(9, 2), 1)]  # 9p played by another player
    state.trick_winner = 1
    task.on_trick_end(state.as_state())
    assert task.status == "fail"

    # Test winning trick without the card
    state.active_cards = [(Card(8, 2), 0)]  # Wrong card
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "fail"

    # Test winning trick with the card
    state.active_cards = [(Card(9, 2), 0)]  # 9p
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "fail"

    # Test failing case - not winning trick with card
    task = AssignedTask(
        formula="9p",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()
    state.active_cards = [(Card(9, 2), 1)]  # 9p played by another player
    state.trick_winner = 1
    task.on_trick_end(state.as_state())
    task.on_game_end()
    assert task.status == "fail"

    # Test failing case - winning trick but wrong card
    task = AssignedTask(
        formula="9p",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()
    state.active_cards = [(Card(8, 2), 0)]  # Wrong card
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    task.on_game_end()
    assert task.status == "fail"


def test_cumulative_tricks():
    """Test '#T>#T(capt)' - must win more tricks than the captain"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="#T>#T(capt)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()
    state.captain = 1  # Set player 1 as captain

    # Player wins first trick
    state.trick = 0
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Captain wins second trick
    state.trick = 1
    state.trick_winner = 1
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Player wins third trick
    state.trick = 2
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    task.on_game_end()
    assert task.status == "success"  # Player won 2 tricks vs captain's 1


def test_cumulative_cards():
    """Test '#p>=5' - must win at least 5 pink cards"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="#p>=5",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Win trick with 2 pink cards
    state.active_cards = [(Card(3, 2), 0), (Card(4, 2), 1)]  # Two pink cards
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Win trick with 2 more pink cards
    state.active_cards = [(Card(5, 2), 0), (Card(6, 2), 1)]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Win trick with 1 more pink card
    state.active_cards = [(Card(7, 2), 0)]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "success"

    task.on_game_end()
    assert task.status == "success"  # Won 5 pink cards total


def test_with_condition():
    """Test '1T with(t)' - must win a trick while playing a trump"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="1T with(t)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Test not winning with trump
    state.active_cards = [(Card(1, 4), 0)]  # Trump card
    state.trick_winner = 1  # But didn't win
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Test winning without trump
    state.active_cards = [(Card(9, 0), 0)]  # Non-trump card
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Test winning with trump
    state.active_cards = [(Card(4, 4), 0)]  # Trump card
    state.trick_winner = 0
    task.on_trick_end(state.as_state())
    assert task.status == "success"


def test_sweep_condition():
    """Test '#sweep>=1' - must win all cards of at least one suit"""
    settings = Settings(side_suit_length=3, task_idxs=(0,))
    task = AssignedTask(
        formula="#sweep>=1",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Win all blue cards (suit 0)
    state.active_cards = [(Card(1, 0), 0)]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())

    state.active_cards = [(Card(2, 0), 0)]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())

    state.active_cards = [(Card(3, 0), 0)]
    state.trick_winner = 0
    task.on_trick_end(state.as_state())

    task.on_game_end()
    assert task.status == "success"


def test_no_lead_condition():
    """Test 'nolead(y,p)' - must not lead yellow or pink"""
    settings = Settings(task_idxs=(0,))
    task = AssignedTask(
        formula="nolead(y,p)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state = MockState()

    # Test leading allowed suit
    state.active_cards = [(Card(1, 0), 0)]  # Lead blue
    state.leader = 0
    task.on_trick_end(state.as_state())
    assert task.status == "unresolved"

    # Test leading forbidden suit
    state.active_cards = [(Card(1, 2), 0)]  # Lead pink
    state.leader = 0
    task.on_trick_end(state.as_state())
    assert task.status == "fail"

    # Test new task leading forbidden suit
    task = AssignedTask(
        formula="nolead(y,p)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state.active_cards = [(Card(1, 3), 0)]  # Lead yellow
    state.leader = 0
    task.on_trick_end(state.as_state())
    assert task.status == "fail"
    # Test success case - never lead forbidden suits
    task = AssignedTask(
        formula="nolead(y,p)",
        desc="",
        difficulty=0,
        task_idx=0,
        player=0,
        settings=settings,
    )
    state.active_cards = [(Card(1, 0), 0)]  # Lead blue
    state.leader = 0
    task.on_trick_end(state.as_state())
    task.on_game_end()
    assert task.status == "success"
