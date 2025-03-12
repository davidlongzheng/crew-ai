from __future__ import absolute_import, annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..lib.utils import coerce_string
from .types import TRUMP_SUIT_NUM

if TYPE_CHECKING:
    from .tasks import Task


@dataclass(frozen=True)
class Settings:
    num_players: int = 4
    num_side_suits: int = 4
    use_trump_suit: bool = True
    side_suit_length: int = 9
    trump_suit_length: int = 4
    use_signals: bool = True
    tasks: list[Task] = field(default_factory=list)

    def __post_init__(self):
        assert self.num_side_suits <= TRUMP_SUIT_NUM

    @property
    def num_tricks(self):
        return (
            self.num_side_suits * self.side_suit_length
            + self.use_trump_suit * self.trump_suit_length
        ) // self.num_players

    @property
    def max_hand_size(self):
        return (
            self.num_side_suits * self.side_suit_length
            + self.use_trump_suit * self.trump_suit_length
            - 1
        ) // self.num_players + 1

    @property
    def num_ranks(self):
        return max(
            self.side_suit_length, self.trump_suit_length if self.use_trump_suit else 0
        )

    @property
    def num_suits(self):
        return self.num_side_suits + self.use_trump_suit

    @staticmethod
    def parse_str(txt):
        kwargs = dict([x.split("=") for x in txt.split(",")])
        for k, v in kwargs.items():
            kwargs[k] = coerce_string(v)

        return Settings(**kwargs)


def easy_tasks():
    from .tasks import Task

    defs = ["#T>=1", "#T>=2", "#T>=1"]
    return [Task(f, "") for f in defs]


def easy_settings():
    return Settings(
        num_players=3,
        side_suit_length=4,
        trump_suit_length=2,
        use_signals=False,
        tasks=easy_tasks(),
    )
