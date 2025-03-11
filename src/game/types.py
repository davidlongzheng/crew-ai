from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


ActionType = Literal["play", "signal", "nosignal"]
Phase = Literal["play", "signal", "end"]


@dataclass(frozen=True)
class Action:
    player: int
    type: ActionType
    card: Card | None

    def __str__(self):
        if self.type == "nosignal":
            return f"P{self.player} does not signal."
        return f"P{self.player} {self.type}s {self.card}."

    def __repr__(self):
        return str(self)


SignalValue = Literal["highest", "lowest", "singleton"]


@dataclass(frozen=True)
class Signal:
    card: Card
    value: SignalValue
    trick: int
