from __future__ import absolute_import, annotations

import random

from .settings import Settings
from .state import State
from .tasks import AssignedTask, get_task_defs
from .types import TRUMP_SUIT_NUM, Action, Card, Phase, Signal, SignalValue
from .utils import split_by_suit


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

    def gen_hands(self, rng) -> list[list[Card]]:
        deck = [
            Card(rank, suit)
            for suit in self.settings.get_suits()
            for rank in range(1, self.settings.get_suit_length(suit) + 1)
        ]
        rng.shuffle(deck)

        hands: list[list[Card]] = [[] for _ in range(self.settings.num_players)]
        cur_player = rng.randint(0, self.settings.num_players - 1)
        for card in deck:
            hands[cur_player].append(card)
            cur_player = (cur_player + 1) % self.settings.num_players

        hands = [sorted(hand, key=lambda x: (x.suit, x.rank)) for hand in hands]

        return hands

    def gen_tasks(self, leader: int, rng) -> list[list[AssignedTask]]:
        task_defs = get_task_defs(self.settings.bank)
        if self.settings.task_idxs:
            task_idxs = list(self.settings.task_idxs)
        else:
            assert (
                self.settings.min_difficulty is not None
                and self.settings.max_difficulty is not None
                and self.settings.max_num_tasks is not None
            )
            difficulty = rng.randint(
                self.settings.min_difficulty, self.settings.max_difficulty
            )
            while True:
                task_idxs = []
                cur_difficulty = 0
                while (
                    len(task_idxs) < self.settings.max_num_tasks
                    and cur_difficulty < difficulty
                ):
                    task_idx = rng.randint(0, len(task_defs) - 1)
                    if task_idx in task_idxs:
                        continue
                    task_diff = task_defs[task_idx][2]
                    if cur_difficulty + task_diff <= difficulty:
                        task_idxs.append(task_idx)
                        cur_difficulty += task_diff

                if cur_difficulty == difficulty:
                    break

        assigned_tasks: list[list[AssignedTask]] = [
            [] for _ in range(self.settings.num_players)
        ]
        if self.settings.task_distro in ["fixed", "shuffle"]:
            if self.settings.task_distro == "shuffle":
                rng.shuffle(task_idxs)
                start_player = rng.randint(0, self.settings.num_players - 1)
            else:
                start_player = leader

            for i, _task_idx in enumerate(task_idxs):
                if _task_idx is None:
                    continue
                player = (start_player + i) % self.settings.num_players
                formula, desc, difficulty = task_defs[_task_idx]
                assigned_tasks[player].append(
                    AssignedTask(
                        formula, desc, difficulty, _task_idx, player, self.settings
                    )
                )
        else:
            assert self.settings.task_distro == "random"
            for i, _task_idx in enumerate(task_idxs):
                if _task_idx is None:
                    continue
                player = rng.randint(0, self.settings.num_players - 1)
                formula, desc, difficulty = task_defs[_task_idx]
                assigned_tasks[player].append(
                    AssignedTask(
                        formula, desc, difficulty, _task_idx, player, self.settings
                    )
                )

        return assigned_tasks

    def reset_state(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        hands = self.gen_hands(rng)
        leader = [
            Card(
                rank=self.settings.trump_suit_length,
                suit=TRUMP_SUIT_NUM,
            )
            in hand
            for hand in hands
        ].index(True)
        phase: Phase = "signal" if self.settings.use_signals else "play"
        assigned_tasks = self.gen_tasks(leader, rng)

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
            assigned_tasks=assigned_tasks,
            status="unresolved",
            value=0.0,
        )

    def calc_trick_winner(self, active_cards: list[tuple[Card, int]]) -> int:
        _, winner = max(
            active_cards,
            key=lambda x: (x[0].is_trump, x[0].suit == self.state.lead_suit, x[0].rank),
        )
        return winner

    def skip_to_next_unsignaled(self) -> None:
        while (
            self.state.cur_player != self.state.leader
            and self.state.signals[self.state.cur_player] is not None
        ):
            self.state.cur_player = self.state.get_next_player()

        if self.state.cur_player == self.state.leader:
            self.state.phase = "play"

    def move(self, action: Action) -> float:
        if self.state.phase == "signal":
            assert self.settings.use_signals
            assert action.player == self.state.cur_player
            player_hand = self.state.hands[self.state.cur_player]
            assert action.type in ["signal", "nosignal"], action.type

            if action.type == "signal":
                assert action.card in player_hand, (action.card, player_hand)
                assert not action.card.is_trump, f"Cannot signal trump: {action.card}"
                assert self.state.signals[self.state.cur_player] is None, (
                    f"P{self.state.cur_player} has already signaled"
                )
                matching_suit_cards = [
                    card for card in player_hand if card.suit == action.card.suit
                ]
                matching_suit_cards = sorted(matching_suit_cards, key=lambda x: x.rank)
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
                )
                self.state.signals[self.state.cur_player] = Signal(
                    action.card, value, self.state.trick
                )

            self.state.actions.append(action)
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

            if self.state.get_next_player() == self.state.leader:
                trick_winner = self.calc_trick_winner(self.state.active_cards)
                self.state.trick_winner = trick_winner
                for tasks in self.state.assigned_tasks:
                    for task in tasks:
                        task.on_trick_end(self.state)
                self.state.trick_winner = None
                self.state.past_tricks.append(
                    ([card for card, _ in self.state.active_cards], trick_winner)
                )
                self.state.trick += 1
                self.state.leader = trick_winner
                self.state.cur_player = trick_winner
                self.state.active_cards = []
                if self.settings.use_signals:
                    self.state.phase = "signal"
            else:
                self.state.cur_player = self.state.get_next_player()

            if self.state.trick == self.settings.num_tricks:
                self.state.phase = "end"
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

        prev_value = self.state.value
        avg_tasks_value = sum(
            task.value * task.difficulty
            for tasks in self.state.assigned_tasks
            for task in tasks
        ) / sum(
            task.difficulty for tasks in self.state.assigned_tasks for task in tasks
        )
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
        if self.state.phase == "signal":
            assert self.settings.use_signals
            ret: list[Action] = [
                Action(player=self.state.cur_player, type="nosignal", card=None)
            ]
            if self.state.signals[self.state.cur_player] is not None:
                return ret
            player_hand = [
                card
                for card in self.state.hands[self.state.cur_player]
                if not card.is_trump
            ]
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
