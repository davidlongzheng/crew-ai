from __future__ import absolute_import, annotations

import cpp_game
from game.settings import Settings
from game.state import State
from game.tasks import AssignedTask
from game.types import TRUMP_SUIT_NUM, Action, Card, Event, Phase, Signal, SignalValue
from game.utils import split_by_suit


def rng_shuffle(li, rng: cpp_game.Rng):
    idxs = rng.shuffle_idxs(len(li))
    return [li[idx] for idx in idxs]


class Engine:
    """Crew game engine for one playthrough of the game.

    Example:
    settings = Settings()
    engine = Engine(settings)
    valid_actions = engine.valid_actions() -> Get valid actions.
    engine.move(valid_actions[0])
    """

    def __init__(
        self,
        settings: Settings,
        seed: int | None = None,
    ) -> None:
        self.settings = settings
        self.reset_state(seed=seed)

    def gen_hands(self, rng: cpp_game.Rng) -> list[list[Card]]:
        deck = [
            Card(rank, suit)
            for suit in self.settings.get_suits()
            for rank in range(1, self.settings.get_suit_length(suit) + 1)
        ]
        deck = rng_shuffle(deck, rng)

        hands: list[list[Card]] = [[] for _ in range(self.settings.num_players)]
        cur_player = rng.randint(0, self.settings.num_players - 1)
        for card in deck:
            hands[cur_player].append(card)
            cur_player = (cur_player + 1) % self.settings.num_players

        hands = [sorted(hand, key=lambda x: (x.suit, x.rank)) for hand in hands]

        return hands

    def gen_tasks(self, rng: cpp_game.Rng) -> tuple[list[int], int]:
        if self.settings.task_idxs:
            task_idxs = list(self.settings.task_idxs)
            difficulty = sum(
                self.settings.task_defs[task_idx][2] for task_idx in task_idxs
            )
        else:
            assert (
                self.settings.min_difficulty is not None
                and self.settings.max_difficulty is not None
                and self.settings.max_num_tasks is not None
            )
            if self.settings.difficulty_distro is None:
                difficulty = rng.randint(
                    self.settings.min_difficulty, self.settings.max_difficulty
                )
            else:
                difficulty = rng.choice(
                    list(
                        range(
                            self.settings.min_difficulty,
                            self.settings.max_difficulty + 1,
                        )
                    ),
                    self.settings.difficulty_distro,
                )
            while True:
                task_idxs = []
                cur_difficulty = 0
                while (
                    len(task_idxs) < self.settings.max_num_tasks
                    and cur_difficulty < difficulty
                ):
                    task_idx = rng.randint(0, len(self.settings.task_defs) - 1)
                    if task_idx in task_idxs:
                        continue
                    task_diff = self.settings.task_defs[task_idx][2]
                    if cur_difficulty + task_diff <= difficulty:
                        task_idxs.append(task_idx)
                        cur_difficulty += task_diff

                if cur_difficulty == difficulty:
                    break

        return task_idxs, difficulty

    def assign_tasks(
        self, leader: int, rng: cpp_game.Rng, task_idxs: list[int]
    ) -> list[list[AssignedTask]]:
        assigned_tasks: list[list[AssignedTask]] = [
            [] for _ in range(self.settings.num_players)
        ]
        if self.settings.task_distro in ["fixed", "shuffle"]:
            if self.settings.task_distro == "shuffle":
                task_idxs = rng_shuffle(task_idxs, rng)
                start_player = rng.randint(0, self.settings.num_players - 1)
            else:
                start_player = leader

            for i, _task_idx in enumerate(task_idxs):
                player = (start_player + i) % self.settings.num_players
                formula, desc, difficulty = self.settings.task_defs[_task_idx]
                assigned_tasks[player].append(
                    AssignedTask(
                        formula=formula,
                        desc=desc,
                        difficulty=difficulty,
                        task_idx=_task_idx,
                        player=player,
                        settings=self.settings,
                    )
                )
        else:
            assert self.settings.task_distro == "random"
            for i, _task_idx in enumerate(task_idxs):
                player = rng.randint(0, self.settings.num_players - 1)
                formula, desc, difficulty = self.settings.task_defs[_task_idx]
                assigned_tasks[player].append(
                    AssignedTask(
                        formula=formula,
                        desc=desc,
                        difficulty=difficulty,
                        task_idx=_task_idx,
                        player=player,
                        settings=self.settings,
                    )
                )

        return assigned_tasks

    def reset_state(self, seed: int | None = None) -> None:
        rng = cpp_game.Rng(seed)
        hands = self.gen_hands(rng)
        leader = [
            Card(
                rank=self.settings.trump_suit_length,
                suit=TRUMP_SUIT_NUM,
            )
            in hand
            for hand in hands
        ].index(True)
        phase: Phase = (
            "draft"
            if self.settings.use_drafting
            else "signal"
            if self.settings.use_signals
            else "play"
        )
        task_idxs, difficulty = self.gen_tasks(rng)
        if self.settings.use_drafting:
            assigned_tasks: list[list[AssignedTask]] = [
                [] for _ in range(self.settings.num_players)
            ]
            unassigned_task_idxs = task_idxs.copy()
        else:
            assigned_tasks = self.assign_tasks(leader, rng, task_idxs)
            unassigned_task_idxs = []

        self.state = State(
            num_players=self.settings.num_players,
            phase=phase,
            hands=hands,
            actions=[],
            trick=0,
            leader=leader,
            captain=leader,
            cur_player=leader,
            active_cards=[],
            past_tricks=[],
            signals=[None for _ in range(self.settings.num_players)],
            trick_winner=None,
            history=[],
            task_idxs=task_idxs,
            difficulty=difficulty,
            unassigned_task_idxs=unassigned_task_idxs,
            assigned_tasks=assigned_tasks,
            status="unresolved",
            value=0.0,
        )
        self.state.history.append(
            Event(type="new_trick", phase=self.state.phase, trick=self.state.trick)
        )

    def calc_trick_winner(self, active_cards: list[tuple[Card, int]]) -> int:
        _, winner = max(
            active_cards,
            key=lambda x: (x[0].is_trump, x[0].suit == self.state.lead_suit, x[0].rank),
        )
        return winner

    def num_drafts_left(self):
        tot = self.settings.num_draft_tricks * self.settings.num_players
        num = self.settings.num_players * self.state.trick + self.state.get_turn()
        return tot - num

    def move(self, action: Action) -> float:
        if self.state.phase == "draft":
            assert self.settings.use_drafting
            assert action.player == self.state.cur_player
            assert action.type in ["draft", "nodraft"], action.type

            if action.type == "draft":
                assert action.task_idx in self.state.unassigned_task_idxs
                self.state.unassigned_task_idxs.remove(action.task_idx)
                formula, desc, difficulty = self.settings.task_defs[action.task_idx]
                self.state.assigned_tasks[self.state.cur_player].append(
                    AssignedTask(
                        formula=formula,
                        desc=desc,
                        difficulty=difficulty,
                        task_idx=action.task_idx,
                        player=self.state.cur_player,
                        settings=self.settings,
                    )
                )
                assert len(self.state.unassigned_task_idxs) + sum(
                    len(x) for x in self.state.assigned_tasks
                ) == len(self.state.task_idxs)
            else:
                assert len(self.state.unassigned_task_idxs) < self.num_drafts_left()

            self.state.actions.append(action)
            self.state.history.append(
                Event(type="action", phase=self.state.phase, action=action)
            )
            self.state.cur_player = self.state.get_next_player()
            if self.state.cur_player == self.state.leader:
                self.state.trick += 1
                if self.state.trick == self.settings.num_draft_tricks:
                    assert len(self.state.unassigned_task_idxs) == 0
                    self.state.trick = 0
                    self.state.phase = "signal" if self.settings.use_signals else "play"
                self.state.history.append(
                    Event(
                        type="new_trick", phase=self.state.phase, trick=self.state.trick
                    )
                )

            return 0.0
        elif self.state.phase == "signal":
            assert self.settings.use_signals
            assert action.player == self.state.cur_player
            player_hand = self.state.hands[self.state.cur_player]
            assert action.type == "signal" or (
                self.settings.use_nosignal and action.type == "nosignal"
            ), action.type

            if action.type == "signal":
                assert action.card in player_hand, (action.card, player_hand)
                if not self.settings.cheating_signal:
                    assert not action.card.is_trump, (
                        f"Cannot signal trump: {action.card}"
                    )
                    assert self.state.signals[self.state.cur_player] is None, (
                        f"P{self.state.cur_player} has already signaled"
                    )

                matching_suit_cards = [
                    card for card in player_hand if card.suit == action.card.suit
                ]
                matching_suit_cards = sorted(matching_suit_cards, key=lambda x: x.rank)
                if not self.settings.cheating_signal:
                    assert (
                        action.card == matching_suit_cards[0]
                        or action.card == matching_suit_cards[-1]
                    ), (
                        f"Signal card must be lowest/highest/singleton: {action.card} {matching_suit_cards}"
                    )
                value: SignalValue = (
                    "singleton"
                    if len(matching_suit_cards) == 1
                    else "highest"
                    if action.card == matching_suit_cards[-1]
                    else "lowest"
                    if action.card == matching_suit_cards[0]
                    else "other"
                )
                self.state.signals[self.state.cur_player] = Signal(
                    action.card, value, self.state.trick
                )

            self.state.actions.append(action)
            self.state.history.append(
                Event(type="action", phase=self.state.phase, action=action)
            )
            self.state.cur_player = self.state.get_next_player()
            if self.state.cur_player == self.state.leader:
                self.state.phase = "play"
        elif self.state.phase == "play":
            assert action.player == self.state.cur_player
            player_hand = self.state.hands[self.state.cur_player]
            assert action.card in player_hand, (action.card, player_hand)
            assert action.type == "play", action.type

            if self.state.cur_player != self.state.leader:
                assert action.card.suit == self.state.lead_suit or all(
                    card.suit != self.state.lead_suit for card in player_hand
                ), (action.card, self.state.lead_suit, player_hand)

            player_hand.remove(action.card)
            self.state.active_cards.append((action.card, action.player))
            self.state.actions.append(action)
            self.state.history.append(
                Event(type="action", phase=self.state.phase, action=action)
            )

            if self.state.get_next_player() == self.state.leader:
                trick_winner = self.calc_trick_winner(self.state.active_cards)
                self.state.trick_winner = trick_winner
                for tasks in self.state.assigned_tasks:
                    for task in tasks:
                        task.on_trick_end(self.state)
                self.state.history.append(
                    Event(
                        type="trick_winner",
                        trick=self.state.trick,
                        trick_winner=trick_winner,
                    )
                )
                self.state.trick_winner = None
                self.state.past_tricks.append(
                    ([card for card, _ in self.state.active_cards], trick_winner)
                )
                self.state.trick += 1
                self.state.leader = trick_winner
                self.state.cur_player = trick_winner
                self.state.active_cards = []
                if self.settings.use_signals and not self.settings.single_signal:
                    self.state.phase = "signal"
                self.state.history.append(
                    Event(
                        type="new_trick", phase=self.state.phase, trick=self.state.trick
                    )
                )
            else:
                self.state.cur_player = self.state.get_next_player()

            if self.state.trick == self.settings.num_tricks:
                self.state.phase = "end"
                self.state.history.append(Event(type="game_ended"))
                for tasks in self.state.assigned_tasks:
                    for task in tasks:
                        task.on_game_end()
                        assert task.status != "unresolved", task
        elif self.state.phase == "end":
            raise ValueError("Game has ended!")
        else:
            raise ValueError(f"Unhandled phase {self.state.phase}")

        self.state.status = (
            "success"
            if all(
                task.status == "success"
                for tasks in self.state.assigned_tasks
                for task in tasks
            )
            else "fail"
            if any(
                task.status == "fail"
                for tasks in self.state.assigned_tasks
                for task in tasks
            )
            or self.state.phase == "end"
            else "unresolved"
        )
        if self.state.phase == "end":
            assert self.state.status != "unresolved"

        assert self.state.phase != "draft"
        prev_value = self.state.value
        if self.settings.weight_by_difficulty:
            avg_tasks_value = sum(
                task.value * task.difficulty
                for tasks in self.state.assigned_tasks
                for task in tasks
            ) / sum(
                task.difficulty for tasks in self.state.assigned_tasks for task in tasks
            )
        else:
            task_values = [
                task.value for tasks in self.state.assigned_tasks for task in tasks
            ]
            avg_tasks_value = sum(task_values) / len(task_values)
        assert -1 <= avg_tasks_value <= 1
        win_bonus = (
            self.settings.win_bonus
            if self.state.status == "success"
            else -self.settings.win_bonus
            if self.state.status == "fail"
            else 0
        )
        self.state.value = (avg_tasks_value + win_bonus) / (self.settings.win_bonus + 1)
        assert -1 <= self.state.value <= 1

        reward = self.state.value - prev_value

        return reward

    def valid_actions(self) -> list[Action]:
        if self.state.phase == "draft":
            assert self.settings.use_drafting
            ret: list[Action] = []
            if len(self.state.unassigned_task_idxs) < self.num_drafts_left():
                ret.append(Action(player=self.state.cur_player, type="nodraft"))

            for task_idx in self.state.unassigned_task_idxs:
                ret.append(
                    Action(
                        player=self.state.cur_player, type="draft", task_idx=task_idx
                    )
                )
            return ret
        elif self.state.phase == "signal":
            assert self.settings.use_signals
            ret = []
            if not self.settings.single_signal and not self.settings.cheating_signal:
                ret.append(Action(player=self.state.cur_player, type="nosignal"))
            if (
                self.state.signals[self.state.cur_player] is not None
                and not self.settings.cheating_signal
            ):
                return ret
            player_hand = [
                card
                for card in self.state.hands[self.state.cur_player]
                if not card.is_trump or self.settings.cheating_signal
            ]
            if self.settings.cheating_signal:
                for card in player_hand:
                    ret.append(
                        Action(player=self.state.cur_player, type="signal", card=card)
                    )
            else:
                for sub_hand in split_by_suit(player_hand):
                    if len(sub_hand) == 1:
                        ret.append(
                            Action(
                                player=self.state.cur_player,
                                type="signal",
                                card=sub_hand[0],
                            )
                        )
                    else:
                        ret += [
                            Action(
                                player=self.state.cur_player,
                                type="signal",
                                card=sub_hand[0],
                            ),
                            Action(
                                player=self.state.cur_player,
                                type="signal",
                                card=sub_hand[-1],
                            ),
                        ]
            return ret
        elif self.state.phase == "play":
            player_hand = self.state.hands[self.state.cur_player]

            if self.state.cur_player != self.state.leader:
                matching_suit_cards = [
                    card for card in player_hand if card.suit == self.state.lead_suit
                ]
                if matching_suit_cards:
                    return [
                        Action(player=self.state.cur_player, type="play", card=card)
                        for card in matching_suit_cards
                    ]

            return [
                Action(player=self.state.cur_player, type="play", card=card)
                for card in player_hand
            ]
        elif self.state.phase == "end":
            raise ValueError("Game has ended!")
        else:
            raise ValueError(f"Unhandled phase {self.state.phase}")
