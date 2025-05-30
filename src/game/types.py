from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass

TRUMP_SUIT_NUM = 4
TO_SUIT_LETTER = {0: "b", 1: "g", 2: "p", 3: "y", 4: "t"}
TO_SUIT_NUM = {v: k for k, v in TO_SUIT_LETTER.items()}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    @property
    def is_trump(self):
        return self.suit == TRUMP_SUIT_NUM

    def __post_init__(self):
        assert 0 <= self.suit <= 4
        assert 1 <= self.rank <= 9

    def __str__(self):
        return f"{self.rank}{TO_SUIT_LETTER[self.suit]}"

    def __repr__(self):
        return str(self)


ActionType = Literal["draft", "nodraft", "play", "signal", "nosignal"]
Phase = Literal["draft", "play", "signal", "end"]


@dataclass(frozen=True)
class Action:
    player: int
    type: ActionType
    card: Card | None = None
    task_idx: int | None = None

    def __post_init__(self):
        if self.type in ["nodraft", "nosignal"]:
            assert self.card is None and self.task_idx is None
        elif self.type in ["play", "signal"]:
            assert self.card is not None and self.task_idx is None
        elif self.type == "draft":
            assert self.card is None and self.task_idx is not None
        else:
            raise ValueError(self.type)

    def __str__(self):
        if self.type == "draft":
            return f"P{self.player} drafts {self.task_idx}."
        elif self.type in ["nodraft", "nosignal"]:
            return f"P{self.player} passes."
        return f"P{self.player} {self.type}s {self.card}."

    def short_str(self):
        if self.type == "draft":
            return str(self.task_idx)
        elif self.type in ["nodraft", "nosignal"]:
            return "no"
        return str(self.card)

    def __repr__(self):
        return str(self)


SignalValue = Literal["highest", "lowest", "singleton", "other"]


@dataclass(frozen=True)
class Signal:
    card: Card
    value: SignalValue
    trick: int


@dataclass(frozen=True)
class Event:
    type: Literal["action", "trick_winner", "new_trick", "game_ended"]
    action: Action | None = None
    phase: Phase | None = None
    trick: int | None = None
    trick_winner: int | None = None
