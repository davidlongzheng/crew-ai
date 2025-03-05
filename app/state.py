from dataclasses import dataclass
from typing import Literal

from app.tasks import AssignedTask
from app.types import Action, Card, Phase, Signal


@dataclass
class State:
    phase: Phase
    hands: list[list[Card]]
    actions: list[Action]
    trick: int
    leader: int
    captain: int
    player_turn: int
    active_cards: list[tuple[Card, int]]
    past_tricks: list[tuple[list[Card], int]]
    signals: list[Signal | None]
    trick_winner: int | None
    assigned_tasks: list[list[AssignedTask]]
    status: Literal["success", "fail", "unresolved"]

    @property
    def lead_suit(self) -> int:
        assert len(self.active_cards) > 0
        return self.active_cards[0][0].suit

    def __str__(self):
        return f"""Phase: {self.phase} Trick: {self.trick} Player: {self.player_turn}

Active Cards:
{"\n".join(f"{player} plays {card}" for card, player in self.active_cards)}

Hands:
{"\n".join(f"{'** ' if i == self.player_turn else ''}{i}: {' '.join(map(str, hand))}" for i, hand in enumerate(self.hands))}
"""
