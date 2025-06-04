from dataclasses import dataclass, replace

import numpy as np
from tensordict import TensorDict

from ai.utils import win_rate_by_difficulty
from game.settings import Settings


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
        state["difficulty_distro"] = tuple(difficulty_distro)

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
