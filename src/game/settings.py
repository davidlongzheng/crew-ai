from __future__ import absolute_import, annotations

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, cast

import click
import numpy as np

from game.types import TRUMP_SUIT_NUM, Card
from lib.utils import coerce_string

DEFAULT_PRESET = "all_ns"


@dataclass(frozen=True)
class Settings:
    num_players: int = 4
    num_side_suits: int = 4
    use_trump_suit: bool = True
    side_suit_length: int = 9
    trump_suit_length: int = 4
    use_signals: bool = True
    cheating_signal: bool = False
    single_signal: bool = False

    bank: str = "all"
    # In fixed, tasks are distributed according to the order of tasks,
    # starting from the leader.
    # In shuffle, tasks are shuffled and distributed clockwise starting
    # from a random player.
    # In random, each task is given to a random player.
    task_distro: Literal["fixed", "shuffle", "random"] = "shuffle"
    task_idxs: tuple[int, ...] = field(default_factory=tuple)
    min_difficulty: int | None = 4
    max_difficulty: int | None = 17
    difficulty_distro: tuple[float, ...] | None = None
    max_num_tasks: int | None = 8

    use_drafting: bool = True
    num_draft_tricks: int = 3

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
    weight_by_difficulty: bool = field(default=True, compare=False)

    def __post_init__(self):
        assert self.num_side_suits <= TRUMP_SUIT_NUM

        assert (
            bool(self.task_idxs)
            == (self.min_difficulty is None)
            == (self.max_difficulty is None)
            == (self.max_num_tasks is None)
        )
        if self.difficulty_distro is not None:
            assert self.min_difficulty is not None
            assert self.max_difficulty is not None
            assert (
                len(self.difficulty_distro)
                == self.max_difficulty - self.min_difficulty + 1
            )
            assert np.isclose(sum(self.difficulty_distro), 1.0)

        if self.use_drafting:
            assert self.num_draft_tricks > 0
            assert self.num_draft_tricks * self.num_players >= self.get_max_num_tasks()

        if self.min_difficulty is not None:
            assert self.min_difficulty <= self.max_difficulty

    @cached_property
    def num_cards(self):
        return (
            self.num_side_suits * self.side_suit_length
            + self.use_trump_suit * self.trump_suit_length
        )

    @cached_property
    def num_tricks(self):
        return self.num_cards // self.num_players

    @cached_property
    def max_hand_size(self):
        return (self.num_cards - 1) // self.num_players + 1

    @cached_property
    def task_defs(self):
        from game.tasks import get_task_defs

        return get_task_defs(self.bank)

    @cached_property
    def num_task_defs(self):
        ret = len(self.task_defs)
        assert ret < 127
        return ret

    @cached_property
    def max_num_actions(self):
        ret = self.max_hand_size + self.use_signals
        if self.use_drafting:
            ret = max(ret, self.get_max_num_tasks() + 1)
        return ret

    def get_suit_idx(self, suit):
        if suit < self.num_side_suits:
            return suit
        elif suit == TRUMP_SUIT_NUM and self.use_trump_suit:
            return self.num_side_suits
        else:
            assert False

    def get_card_idx(self, card: Card) -> int:
        return self.get_suit_idx(card.suit) * self.side_suit_length + (card.rank - 1)

    def get_card(self, card_idx: int) -> Card:
        suit = self.get_suit(card_idx // self.side_suit_length)
        rank = (card_idx % self.side_suit_length) + 1
        return Card(rank, suit)

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

    def get_phase_idx(self, phase: str):
        if phase == "play":
            return 0
        elif phase == "signal":
            assert self.use_signals
            return 1
        elif phase == "draft":
            assert self.use_drafting
            return 2 if self.use_signals else 1
        else:
            raise ValueError(phase)

    def get_phase(self, phase_idx: int):
        if phase_idx == 0:
            return "play"
        elif self.use_signals and phase_idx == 1:
            return "signal"
        else:
            assert self.use_drafting and (
                (phase_idx == 1 and not self.use_signals)
                or (phase_idx == 2 and self.use_signals)
            )
            return "draft"

    @cached_property
    def use_nosignal(self):
        return self.use_signals and not self.single_signal and not self.cheating_signal

    @cached_property
    def max_suit_length(self):
        return max(
            self.side_suit_length, self.trump_suit_length if self.use_trump_suit else 0
        )

    @cached_property
    def num_suits(self):
        return self.num_side_suits + self.use_trump_suit

    @cached_property
    def num_phases(self):
        return 1 + self.use_signals + self.use_drafting

    def get_max_num_tasks(self) -> int:
        if self.task_idxs:
            return len(self.task_idxs)
        else:
            return cast(int, self.max_num_tasks)

    def get_seq_length(self) -> int:
        return self.num_players * (
            self.num_tricks * (2 if self.use_signals and not self.single_signal else 1)
            + self.single_signal
            + self.use_drafting * self.num_draft_tricks
        )

    def to_cpp(self):
        import cpp_game

        cpp_settings = cpp_game.Settings(
            self.num_players,
            self.num_side_suits,
            self.use_trump_suit,
            self.side_suit_length,
            self.trump_suit_length,
            self.use_signals,
            self.cheating_signal,
            self.single_signal,
            self.bank,
            self.task_distro,
            self.task_idxs,
            self.min_difficulty,
            self.max_difficulty,
            self.difficulty_distro,
            self.max_num_tasks,
            self.use_drafting,
            self.num_draft_tricks,
            self.task_bonus,
            self.win_bonus,
            self.weight_by_difficulty,
        )
        return cpp_settings


def get_preset(preset):
    if preset == "easy":
        return Settings(
            bank="easy",
            use_signals=False,
            min_difficulty=None,
            max_difficulty=None,
            max_num_tasks=None,
            use_drafting=False,
            task_idxs=(0, 0, 1, 2),
        )
    elif preset == "all_ns":
        return Settings(
            use_signals=False,
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

        preset = get_preset(kwargs.pop("preset", DEFAULT_PRESET))
        return dataclasses.replace(preset, **kwargs)


SETTINGS_TYPE = SettingsType()
