from collections import Counter
from functools import cache

import pandas as pd

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Action
from ..lib.types import StrMap, req

HIST_ONLY = False


def set_aux_info_hist_only(x):
    global HIST_ONLY
    HIST_ONLY = x


@cache
def get_aux_info_spec(settings: Settings) -> pd.DataFrame:
    """If num_cat=None, then it's a continuous variable."""
    spec: list[tuple[str, int | float, float]] = []
    # 3
    spec += [
        (f"hist_tricks_p{player_idx}", float("nan"), 5.0)
        for player_idx in range(settings.num_players)
    ]
    # 5
    spec += [
        (f"cards_left_s{suit_idx}", float("nan"), 1.0)
        for suit_idx in range(settings.num_suits)
    ]
    # 5
    spec += [
        (f"max_rank_left_s{suit_idx}", float("nan"), 1.0)
        for suit_idx in range(settings.num_suits)
    ]
    # 5
    spec += [
        (f"min_rank_left_s{suit_idx}", float("nan"), 1.0)
        for suit_idx in range(settings.num_suits)
    ]
    # 3
    spec += [
        (f"hist_frac_success_p{player_idx}", float("nan"), 1.0)
        for player_idx in range(settings.num_players)
    ]
    if not HIST_ONLY:
        # 3
        spec += [
            (f"fut_tricks_p{player_idx}", float("nan"), 5.0)
            for player_idx in range(settings.num_players)
        ]
        # 3
        spec += [
            (f"frac_success_p{player_idx}", float("nan"), 1.0)
            for player_idx in range(settings.num_players)
        ]

    # I know this isn't HIST_ONLY but whatevs.
    # 3
    spec.append(("won_cur_trick_pidx", settings.num_players, 0.1))

    return pd.DataFrame(spec, columns=["name", "num_cat", "weight"])


class AuxInfoTracker:
    def __init__(self, settings: Settings, engine: Engine):
        self.settings = settings
        self.aux_info_spec = get_aux_info_spec(settings)
        self.num_players = settings.num_players
        self.engine = engine
        self.aux_info_pt: list[StrMap] = []
        self.num_tricks_won: Counter = Counter()
        self.cur_trick = 0
        self.cur_trick_first_move = 0
        self.cards_unplayed: dict[int, set[int]] = {
            suit: set(range(1, self.settings.get_suit_length(suit) + 1))
            for suit in self.settings.get_suits()
        }

    @property
    def cur_move(self):
        return len(self.aux_info_pt)

    def get_aux_info(self):
        """Return a list of lists of normalized values."""

        def transform(x):
            return [x[name] for name in self.aux_info_spec["name"]]

        return [transform(x) for x in self.aux_info_pt]

    def get_frac_success(self):
        ret = {}
        for player_idx in range(self.num_players):
            player = self.engine.state.get_player(player_idx)
            tasks = self.engine.state.assigned_tasks[player]
            ret[player_idx] = sum(x.status == "success" for x in tasks) / max(
                len(tasks), 1
            )
        return ret

    def on_decision(self):
        aux_info = {}

        frac_success = self.get_frac_success()
        for player_idx in range(self.num_players):
            aux_info[f"hist_tricks_p{player_idx}"] = (
                self.num_tricks_won[player_idx] / self.settings.num_tricks
            )
            aux_info[f"hist_frac_success_p{player_idx}"] = frac_success[player_idx]

        for suit, cards_unplayed in self.cards_unplayed.items():
            suit_idx = self.settings.get_suit_idx(suit)
            suit_length = self.settings.get_suit_length(suit)
            aux_info[f"cards_left_s{suit_idx}"] = len(cards_unplayed) / suit_length
            aux_info[f"max_rank_left_s{suit_idx}"] = (
                max(cards_unplayed, default=0) / suit_length
            )
            aux_info[f"min_rank_left_s{suit_idx}"] = (
                min(cards_unplayed, default=suit_length + 1) - 1
            ) / suit_length

        self.aux_info_pt.append(aux_info)

    def on_move(self, action: Action):
        if action.type == "play":
            card = req(action.card)
            self.cards_unplayed[card.suit].remove(card.rank)

        # Handle end of trick
        if self.engine.state.trick > self.cur_trick:
            assert len(self.engine.state.past_tricks) == self.engine.state.trick
            last_trick = self.engine.state.past_tricks[-1]
            _, trick_winner = last_trick
            trick_winner_pidx = self.engine.state.get_player_idx(trick_winner)

            for move in range(self.cur_trick_first_move, self.cur_move):
                self.aux_info_pt[move]["won_cur_trick_pidx"] = trick_winner_pidx

            self.num_tricks_won[trick_winner_pidx] += 1
            self.cur_trick_first_move = self.cur_move
            self.cur_trick += 1

    def on_game_end(self):
        frac_success = self.get_frac_success()
        for player_idx in range(self.num_players):
            for aux_info in self.aux_info_pt:
                aux_info[f"fut_tricks_p{player_idx}"] = (
                    self.num_tricks_won[player_idx] / self.settings.num_tricks
                    - aux_info[f"hist_tricks_p{player_idx}"]
                )
                aux_info[f"frac_success_p{player_idx}"] = frac_success[player_idx]
