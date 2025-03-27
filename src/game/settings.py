from __future__ import absolute_import, annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, cast

import click

from ..lib.utils import coerce_string
from .types import TRUMP_SUIT_NUM


@dataclass(frozen=True)
class Settings:
    num_players: int = 4
    num_side_suits: int = 4
    use_trump_suit: bool = True
    side_suit_length: int = 9
    trump_suit_length: int = 4
    use_signals: bool = True

    bank: str = "easy"
    # In fixed, tasks are distributed according to the order of tasks,
    # starting from the leader.
    # In shuffle, tasks are shuffled and distributed clockwise starting
    # from a random player.
    # In random, each task is given to a random player.
    task_distro: Literal["fixed", "shuffle", "random"] = "shuffle"
    task_idxs: tuple[int | None, ...] = field(default_factory=tuple)
    min_difficulty: int | None = None
    max_difficulty: int | None = None
    max_num_tasks: int | None = None

    # Task bonus is how much of a bonus you get for a fully completed task
    # vs a partially completed task.
    # Partial points range from [-1, 1] and task bonus is {-task_bonus, 0, task_bonus}
    # depending on if the task was fail/unresolved/success.
    # Then everything is rescaled by 1 / (1 + task_bonus) so that the final value
    # ranges from [-1, 1].
    task_bonus: float = field(default=5, compare=False)
    # Win bonus is like task bonus but for a game win vs a partially
    # completed tasks. The math is the same s.t. the final value is again
    # from [-1, 1]
    win_bonus: float = field(default=1, compare=False)

    def __post_init__(self):
        assert self.num_side_suits <= TRUMP_SUIT_NUM

        assert (
            bool(self.task_idxs)
            == (self.min_difficulty is None)
            == (self.max_difficulty is None)
            == (self.max_num_tasks is None)
        )

        if self.min_difficulty is not None:
            assert self.min_difficulty < self.max_difficulty

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

    def get_suit_idx(self, suit):
        if suit < self.num_side_suits:
            return suit
        elif suit == TRUMP_SUIT_NUM and self.use_trump_suit:
            return self.num_side_suits
        else:
            assert False

    def get_suit(self, suit_idx):
        if suit_idx < self.num_side_suits:
            return suit_idx
        elif suit_idx == self.num_side_suits and self.use_trump_suit:
            return TRUMP_SUIT_NUM
        else:
            assert False

    def get_suits(self):
        ret = list(range(self.num_side_suits))
        if self.use_trump_suit:
            ret.append(TRUMP_SUIT_NUM)
        return ret

    def get_suit_length(self, suit):
        if suit < self.num_side_suits:
            return self.side_suit_length
        elif suit == TRUMP_SUIT_NUM and self.use_trump_suit:
            return self.trump_suit_length
        else:
            assert False

    @property
    def max_suit_length(self):
        return max(
            self.side_suit_length, self.trump_suit_length if self.use_trump_suit else 0
        )

    @property
    def num_suits(self):
        return self.num_side_suits + self.use_trump_suit

    @property
    def num_phases(self):
        return 1 + self.use_signals

    def get_max_num_tasks(self) -> int:
        if self.task_idxs:
            return sum(x is not None for x in self.task_idxs)
        else:
            return cast(int, self.max_num_tasks)


def get_preset(preset):
    if preset == "easy_p3":
        return Settings(
            num_players=3,
            side_suit_length=4,
            trump_suit_length=2,
            use_signals=False,
            bank="easy",
            task_idxs=(0, 0, 1),
        )
    elif preset == "easy_p4":
        return Settings(
            use_signals=False,
            bank="easy",
            task_idxs=(0, 0, 1, 1),
        )
    elif preset == "med":
        return Settings(
            use_signals=False,
            bank="med",
            min_difficulty=1,
            max_difficulty=3,
            max_num_tasks=4,
        )
    else:
        raise ValueError(preset)


class SettingsType(click.ParamType):
    name = "Settings"

    def convert(self, value, param, ctx):
        if isinstance(value, Settings):
            return value

        kwargs = dict([x.split("=") for x in value.split(",")])

        for k, v in kwargs.items():
            kwargs[k] = coerce_string(v)

        preset = get_preset(kwargs.pop("preset", "easy_p3"))
        return dataclasses.replace(preset, **kwargs)


SETTINGS_TYPE = SettingsType()
