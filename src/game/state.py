from __future__ import absolute_import, annotations

from typing import Literal

from pydantic.dataclasses import dataclass

from game.tasks import AssignedTask
from game.types import Action, Card, Event, Phase, Signal


@dataclass
class State:
    num_players: int
    phase: Phase
    hands: list[list[Card]]
    actions: list[Action]
    trick: int
    leader: int
    captain: int
    cur_player: int
    active_cards: list[tuple[Card, int]]
    past_tricks: list[tuple[list[Card], int]]
    signals: list[Signal | None]
    trick_winner: int | None
    history: list[Event]
    task_idxs: list[int]
    difficulty: int
    unassigned_task_idxs: list[int]
    assigned_tasks: list[list[AssignedTask]]
    status: Literal["success", "fail", "unresolved"]
    value: float
    shown_out: list[dict[int, bool]]

    def get_next_player(self, player=None):
        player = self.cur_player if player is None else player
        return (player + 1) % self.num_players

    def get_player_idx(self, player=None):
        player = self.cur_player if player is None else player
        return (player - self.captain) % self.num_players

    def get_player(self, player_idx):
        return (self.captain + player_idx) % self.num_players

    def get_turn(self, player=None):
        player = self.cur_player if player is None else player
        return (player - self.leader) % self.num_players

    @property
    def lead_suit(self) -> int:
        assert len(self.active_cards) > 0
        return self.active_cards[0][0].suit

    def __str__(self):
        return f"""Phase: {self.phase} Trick: {self.trick} Player: {self.cur_player}

Active Cards:
{"\n".join(f"{player} plays {card}" for card, player in self.active_cards)}

Hands:
{"\n".join(f"{'** ' if i == self.cur_player else ''}{i}: {' '.join(map(str, hand))}" for i, hand in enumerate(self.hands))}
"""
