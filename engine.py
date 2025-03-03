import attrs
import random


@attrs.frozen()
class Card:
    rank: int
    suit: int
    is_trump: bool


class Engine:
    """Crew game engine for one playthrough of the game.

    Example:
    game = Game()
    game.reset() -> Reset all state
    game.global_state() -> Get global state
    game.cur_player_state() -> Get state of current player.
    game.valid_actions() -> Get valid actions.
    game.play(my_action) -> Returns (is_legal, output)
    """

    def __init__(
        self,
        num_players=4,
        num_side_suits=4,
        use_trump_suit=True,
        side_suit_length=9,
        trump_suit_length=4,
        num_signals=1,
    ):
        self.num_players = num_players
        self.num_side_suits = num_side_suits
        self.use_trump_suit = use_trump_suit
        self.side_suit_length = side_suit_length
        self.trump_suit_length = trump_suit_length
        self.num_signals = num_signals
        self.reset()

    def gen_hands(self):
        deck = [
            Card(rank, suit, is_trump=False)
            for rank in range(self.num_side_suits)
            for suit in range(self.side_suit_length)
        ] + [
            Card(rank, self.num_side_suits, is_trump=True)
            for rank in range(self.trump_suit_length)
        ]

        random.shuffle(deck)

    def reset(self):
        self.phase = "play"
        self.actions = []
        self.trick = 0
        self.turn = 0
        self.leader = 0
        self.hands = self.gen_hands()
