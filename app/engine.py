import random

from app.settings import Settings
from app.state import State
from app.tasks import AssignedTask
from app.types import TRUMP_SUIT_NUM, Action, Card, Signal, SignalValue, Phase
from app.utils import split_by_suit


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

    def gen_hands(self, rng: random.Random | None = None) -> list[list[Card]]:
        if not rng:
            rng = random.Random()

        deck = [
            Card(rank, suit)
            for rank in range(1, self.settings.side_suit_length + 1)
            for suit in range(self.settings.num_side_suits)
        ]
        if self.settings.use_trump_suit:
            deck += [
                Card(rank, TRUMP_SUIT_NUM)
                for rank in range(1, self.settings.trump_suit_length + 1)
            ]
        rng.shuffle(deck)

        hands: list[list[Card]] = [[] for _ in range(self.settings.num_players)]
        cur_player = rng.randint(0, self.settings.num_players - 1)
        for card in deck:
            hands[cur_player].append(card)
            cur_player = (cur_player + 1) % self.settings.num_players

        hands = [sorted(hand, key=lambda x: (x.suit, x.rank)) for hand in hands]

        return hands

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

        assigned_tasks: list[list[AssignedTask]] = [
            [] for _ in range(self.settings.num_players)
        ]
        for i, task in enumerate(self.settings.tasks):
            assigned_player = i % self.settings.num_players
            assigned_tasks[assigned_player].append(
                AssignedTask(task.formula, assigned_player, self.settings)
            )

        self.state = State(
            phase=phase,
            hands=hands,
            actions=[],
            trick=0,
            leader=leader,
            captain=leader,
            player_turn=leader,
            active_cards=[],
            past_tricks=[],
            signals=[None for _ in range(self.settings.num_players)],
            trick_winner=None,
            assigned_tasks=assigned_tasks,
            status="unresolved",
        )

    def calc_trick_winner(self, active_cards: list[tuple[Card, int]]) -> int:
        _, winner = max(
            active_cards,
            key=lambda x: (x[0].is_trump, x[0].suit == self.state.lead_suit, x[0].rank),
        )
        return winner

    def skip_to_next_unsignaled(self) -> None:
        while (
            self.state.player_turn != self.state.leader
            and self.state.signals[self.state.player_turn] is not None
        ):
            self.state.player_turn = (
                self.state.player_turn + 1
            ) % self.settings.num_players

        if self.state.player_turn == self.state.leader:
            self.state.phase = "play"

    def move(self, action: Action) -> None:
        if self.state.phase == "signal":
            assert self.settings.use_signals
            assert action.player == self.state.player_turn
            player_hand = self.state.hands[self.state.player_turn]
            assert action.type in ["signal", "nosignal"], action.type

            if action.type == "signal":
                assert action.card in player_hand, (action.card, player_hand)
                assert not action.card.is_trump, f"Cannot signal trump: {action.card}"
                assert self.state.signals[self.state.player_turn] is None, (
                    f"P{self.state.player_turn} has already signaled"
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
                self.state.signals[self.state.player_turn] = Signal(
                    action.card, value, self.state.trick
                )

            self.state.actions.append(action)
            self.state.player_turn = (
                self.state.player_turn + 1
            ) % self.settings.num_players
            self.skip_to_next_unsignaled()
        elif self.state.phase == "play":
            assert action.player == self.state.player_turn
            player_hand = self.state.hands[self.state.player_turn]
            assert action.card in player_hand, (action.card, player_hand)
            assert action.type == "play", action.type

            if self.state.player_turn != self.state.leader:
                assert action.card.suit == self.state.lead_suit or all(
                    card.suit != self.state.lead_suit for card in player_hand
                ), (action.card, self.state.lead_suit, player_hand)

            player_hand.remove(action.card)
            self.state.active_cards.append((action.card, action.player))
            self.state.actions.append(action)

            if (
                self.state.player_turn + 1
            ) % self.settings.num_players == self.state.leader:
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
                self.state.player_turn = trick_winner
                self.state.active_cards = []
                if self.settings.use_signals:
                    self.skip_to_next_unsignaled()
            else:
                self.state.player_turn = (
                    self.state.player_turn + 1
                ) % self.settings.num_players

            if self.state.trick == self.settings.num_tricks:
                self.state.phase = "end"
                for tasks in self.state.assigned_tasks:
                    for task in tasks:
                        task.on_game_end()
                        assert task.status != "unresolved", task
                self.state.status = (
                    "success"
                    if all(
                        task.status == "success"
                        for tasks in self.state.assigned_tasks
                        for task in tasks
                    )
                    else "fail"
                )

        elif self.state.phase == "end":
            raise ValueError("Game has ended!")
        else:
            raise ValueError(f"Unhandled phase {self.state.phase}")

    def valid_actions(self) -> list[Action]:
        if self.state.phase == "signal":
            assert self.settings.use_signals
            assert self.state.signals[self.state.player_turn] is None
            player_hand = [
                card
                for card in self.state.hands[self.state.player_turn]
                if not card.is_trump
            ]
            ret: list[Action] = [
                Action(player=self.state.player_turn, type="nosignal", card=None)
            ]
            for sub_hand in split_by_suit(player_hand):
                if len(sub_hand) == 1:
                    ret.append(
                        Action(
                            player=self.state.player_turn,
                            type="signal",
                            card=sub_hand[0],
                        )
                    )
                else:
                    ret += [
                        Action(
                            player=self.state.player_turn,
                            type="signal",
                            card=sub_hand[0],
                        ),
                        Action(
                            player=self.state.player_turn,
                            type="signal",
                            card=sub_hand[-1],
                        ),
                    ]
            return ret
        elif self.state.phase == "play":
            player_hand = self.state.hands[self.state.player_turn]

            if self.state.player_turn != self.state.leader:
                matching_suit_cards = [
                    card for card in player_hand if card.suit == self.state.lead_suit
                ]
                if matching_suit_cards:
                    return [
                        Action(player=self.state.player_turn, type="play", card=card)
                        for card in matching_suit_cards
                    ]

            return [
                Action(player=self.state.player_turn, type="play", card=card)
                for card in player_hand
            ]
        elif self.state.phase == "end":
            raise ValueError("Game has ended!")
        else:
            raise ValueError(f"Unhandled phase {self.state.phase}")
