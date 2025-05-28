from dataclasses import dataclass, replace

import numpy as np
from tensordict import TensorDict

from ai.utils import win_rate_by_difficulty
from game.settings import Settings


@dataclass(frozen=True)
class Lesson:
    # difficulty
    min_difficulty: int
    max_difficulty: int
    difficulty_distro: list[float] | None = None

    # triggers for advancing to next lesson
    target_difficulty: int | None = None
    # Minimum win rate on target difficulty before advancing
    win_thresh: float | None = None
    no_improve_num_rounds: int | None = None
    min_improve_by: float = 0.0

    def __post_init__(self):
        assert self.min_difficulty <= self.max_difficulty

        if self.difficulty_distro is not None:
            assert len(self.difficulty_distro) == (
                self.max_difficulty - self.min_difficulty + 1
            )

        if self.win_thresh is not None or self.no_improve_num_rounds is not None:
            assert (
                self.target_difficulty is not None
                and self.min_difficulty <= self.target_difficulty <= self.max_difficulty
            )

    def should_advance(self, td, lesson_state: dict):
        win_rate = td["win"][td["difficulty"] == self.target_difficulty].float().mean()

        if (
            "last_win_rate" not in lesson_state
            or win_rate > lesson_state["last_win_rate"] + self.min_improve_by
        ):
            lesson_state["last_win_rate"] = win_rate.item()
            lesson_state["no_improve_num_rounds"] = 0
        else:
            lesson_state["no_improve_num_rounds"] += 1

        if self.win_thresh and win_rate < self.win_thresh:
            return False

        if (
            self.no_improve_num_rounds
            and lesson_state["no_improve_num_rounds"] < self.no_improve_num_rounds
        ):
            return False

        return True


@dataclass(frozen=True)
class Curriculum:
    start_min_difficulty: int = 4
    start_max_difficulty: int = 7
    end_max_difficulty: int = 17
    # Win rate in which we deem the difficulty "cleared"
    win_rate_to_clear: float = 0.8
    # Win rate for which we phase in the next difficulty level.
    win_rate_to_next_level: float = 0.4
    cleared_prob: float = 0.05
    cleared_total_prob: float = 0.20

    def update_settings(
        self, state: dict, settings: Settings, td: TensorDict | None = None
    ) -> Settings | None:
        state.setdefault("min_target_difficulty", self.start_min_difficulty)
        state.setdefault("max_difficulty", self.start_max_difficulty)
        state.setdefault("difficulty_distro", None)

        if td is None:
            return replace(
                settings,
                min_difficulty=self.start_min_difficulty,
                max_difficulty=state["max_difficulty"],
                difficulty_distro=state["difficulty_distro"],
            )

        win_rates = win_rate_by_difficulty(td)
        new_settings = False
        while win_rates[state["min_target_difficulty"]] >= self.win_rate_to_clear:
            state["min_target_difficulty"] += 1
            new_settings = True

        if win_rates[state["max_difficulty"]] >= self.win_rate_to_next_level:
            state["max_difficulty"] += 1
            new_settings = True

        if not new_settings:
            return None

        n_cleared = state["min_target_difficulty"] - self.start_min_difficulty
        difficulty_distro = [self.cleared_prob] * n_cleared
        if sum(difficulty_distro) > self.cleared_total_prob:
            difficulty_distro_arr = np.array(difficulty_distro)
            difficulty_distro = list(
                difficulty_distro_arr
                / np.sum(difficulty_distro_arr)
                * self.cleared_total_prob
            )

        n_uncleared = state["max_difficulty"] - state["min_target_difficulty"] + 1
        uncleared_prob = (1.0 - sum(difficulty_distro)) / n_uncleared
        difficulty_distro += [uncleared_prob] * n_uncleared
        state["difficulty_distro"] = difficulty_distro

        return replace(
            settings,
            min_difficulty=self.start_min_difficulty,
            max_difficulty=state["max_difficulty"],
            difficulty_distro=state["difficulty_distro"],
        )


def get_curriculum(curriculum_mode):
    if curriculum_mode == "v2":
        return Curriculum()
    else:
        raise ValueError(curriculum_mode)
