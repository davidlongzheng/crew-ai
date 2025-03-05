from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, cast, get_args

from app.settings import Settings
from app.types import TO_SUIT_NUM, Card

if TYPE_CHECKING:
    from app.state import State

Status = Literal["success", "fail", "unresolved"]
Direction = Literal[">", "<", ">=", "=", "<="]
CardFilter = Callable[[Card], bool]
TrickCountType = Literal["capt", "sumothers", "anyother"]


@dataclass(kw_only=True)
class Condition:
    settings: Settings
    player: int
    status: Status = "unresolved"

    def reset(self):
        pass

    def on_trick_end(self, state: State):
        pass

    def on_game_end(self):
        pass


@dataclass
class TrickCond(Condition):
    trick: int
    no: bool = False

    def reset(self):
        self.status = "unresolved"

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick == self.trick:
            won_trick = state.trick_winner == self.player
            self.status = "success" if won_trick == (not self.no) else "fail"

    def on_game_end(self):
        assert self.status != "unresolved"


@dataclass
class CardCond(Condition):
    card: Card

    def reset(self):
        self.status = "unresolved"

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player and any(
            self.card == card for card, _ in state.active_cards
        ):
            self.status = "success"

    def on_game_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class CumTrickCond(Condition):
    direction: Direction
    num_tricks: int | TrickCountType

    num_tricks_won: int = 0
    num_other_tricks_won: int = 0
    num_other_tricks_won_per_player: list[int] = field(default_factory=list)

    def reset(self):
        self.status = "unresolved"
        self.num_tricks_won = 0
        if self.num_tricks == "anyother":
            self.num_other_tricks_won_per_player = [
                0 for _ in range(self.settings.num_players)
            ]
        else:
            self.num_other_tricks_won = 0

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            self.num_tricks_won += 1
        else:
            if self.num_tricks == "capt":
                assert state.captain != self.player
                if state.trick_winner == state.captain:
                    self.num_other_tricks_won += 1
            elif self.num_tricks == "sumothers":
                self.num_other_tricks_won += 1
            elif self.num_tricks == "anyothers":
                self.num_other_tricks_won_per_player[state.trick_winner] += 1
            else:
                assert isinstance(self.num_tricks, int)

    def on_game_end(self):
        if isinstance(self.num_tricks, int):
            num_other_tricks_won = self.num_tricks
        elif self.num_tricks in {"capt", "sumothers"}:
            num_other_tricks_won = self.num_other_tricks_won
        else:
            assert self.num_tricks == "anyother"
            num_other_tricks_won_per_player = [
                x
                for i, x in enumerate(self.num_other_tricks_won_per_player)
                if i != self.player
            ]
            num_other_tricks_won = (
                max(num_other_tricks_won_per_player)
                if self.direction in [">=", ">"]
                else min(num_other_tricks_won_per_player)
            )

        self.status = (
            "success"
            if compare_dir(self.num_tricks_won, num_other_tricks_won, self.direction)
            else "fail"
        )


@dataclass
class CumCardCond(Condition):
    direction: Direction
    card_filter: CardFilter
    num_cards: int | None = None
    other_card_filter: CardFilter | None = None

    # Track cards won that match each filter
    cards_won: int = 0
    other_cards_won: int = 0

    def reset(self):
        self.status = "unresolved"
        self.cards_won = 0
        self.other_cards_won = 0

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            # Count cards in this trick that match our filters
            self.cards_won += sum(
                1 for card, _ in state.active_cards if self.card_filter(card)
            )
            if self.other_card_filter:
                self.other_cards_won += sum(
                    1 for card, _ in state.active_cards if self.other_card_filter(card)
                )

    def on_game_end(self):
        compare_value = (
            self.num_cards if self.num_cards is not None else self.other_cards_won
        )
        self.status = (
            "success"
            if compare_dir(self.cards_won, compare_value, self.direction)
            else "fail"
        )


@dataclass
class WithCond(Condition):
    card_filter: CardFilter

    def reset(self):
        self.status = "unresolved"

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            (player_card,) = [
                card for card, player in state.active_cards if player == self.player
            ]
            # Check if any card in the trick matches our filter
            if self.card_filter(player_card):
                self.status = "success"

    def on_game_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class SweepCond(Condition):
    num_sweeps: int

    # Track cards won per suit
    cards_won_per_suit: dict[int, int] = field(default_factory=dict)

    def reset(self):
        self.status = "unresolved"
        self.cards_won_per_suit = {i: 0 for i in range(self.settings.num_side_suits)}

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            # Add all cards from this trick to their respective suit sets
            for card, _ in state.active_cards:
                if not card.is_trump:
                    self.cards_won_per_suit[card.suit] += 1

    def on_game_end(self):
        num_sweeps = sum(
            x == self.settings.side_suit_length
            for x in self.cards_won_per_suit.values()
        )
        self.status = "success" if num_sweeps >= self.num_sweeps else "fail"


@dataclass
class ConsecCond(Condition):
    num_consec: int
    no: bool = False

    won_tricks: set[int] = field(default_factory=set)

    def __post_init__(self):
        assert self.num_consec >= 2

    def reset(self):
        self.status = "unresolved"
        self.won_tricks = set()

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            self.won_tricks.add(state.trick)

    def on_game_end(self):
        cur_consec_start, cur_consec_end = None, None
        best_consec_start, best_consec_end = None, None

        for trick in self.won_tricks:
            if cur_consec_end is not None and trick == cur_consec_end + 1:
                cur_consec_end = trick
            else:
                cur_consec_start = cur_consec_end = trick

            if cur_consec_start is not None and (
                best_consec_start is None
                or (cur_consec_end - cur_consec_start + 1)
                > (best_consec_end - best_consec_start + 1)
            ):
                best_consec_start = cur_consec_start
                best_consec_end = cur_consec_end

        best_consec = (
            0
            if best_consec_start is None
            else (best_consec_end - best_consec_start + 1)
        )
        if (best_consec >= self.num_consec) == (not self.no):
            self.status = "success"
        else:
            self.status = "fail"


@dataclass
class SumCond(Condition):
    direction: Direction
    sum: int

    def reset(self):
        self.status = "unresolved"

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            trick_sum = sum(card.rank for card, _ in state.active_cards)
            if compare_dir(trick_sum, self.sum, self.direction):
                self.status = "success"

    def on_game_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class NoLeadCond(Condition):
    suits: list[int]

    def reset(self):
        self.status = "unresolved"

    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        # If we're the leader and we lead a forbidden suit, fail
        if state.leader == self.player and state.active_cards[0][0].suit in self.suits:
            self.status = "fail"

    def on_game_end(self):
        if self.status == "unresolved":
            self.status = "success"


def parse_dir(token: str) -> tuple[Direction, str]:
    if token.startswith("<=") or token.startswith(">="):
        return cast(Direction, token[:2]), token[2:]
    assert token.startswith(">") or token.startswith("<") or token.startswith("=")
    return cast(Direction, token[0]), token[1:]


def parse_card(token: str) -> tuple[Card | None, str]:
    if token[0].isdigit() and token[1] in TO_SUIT_NUM:
        rank = int(token[0])
        suit = TO_SUIT_NUM[token[1]]
        token = token[2:]
        return Card(rank, suit), token
    else:
        return None, token


def compare_dir(a: int, b: int, dir: Direction) -> bool:
    if dir == ">=":
        return a >= b
    elif dir == "<=":
        return a <= b
    elif dir == ">":
        return a > b
    elif dir == "<":
        return a < b
    elif dir == "=":
        return a == b
    else:
        raise ValueError(dir)


def parse_card_filter(token: str) -> tuple[CardFilter | None, str]:
    if token.startswith("odd") or token.startswith("even"):
        pfx = "odd" if token.startswith("odd") else "even"
        token = token.removeprefix(pfx)

        def filt(card: Card) -> bool:
            if pfx == "odd":
                return card.rank % 2 == 1
            else:
                return card.rank % 2 == 0

        return filt, token
    elif token.startswith("rank("):
        token = token.removeprefix("rank(")
        assert token.count(")") == 1
        rank_filter, token = token.split(")")
        direction, num = parse_dir(rank_filter)

        def filt(card: Card) -> bool:
            return compare_dir(card.rank, int(num), direction)

        return filt, token
    else:
        rank: int | None = None
        if token and token[0].isdigit():
            rank = int(token[0])
            token = token[1:]

        suit: int | None = None
        if token and token[0] in TO_SUIT_NUM:
            suit = TO_SUIT_NUM[token[0]]
            token = token[1:]

        if rank is None and suit is None:
            return None, token

        def filt(card: Card) -> bool:
            return (rank is None or rank == card.rank) and (
                suit is None or suit == card.suit
            )

        return filt, token


def parse_token(token: str, settings: Settings, player: int) -> Condition:
    orig_token = token

    if token.startswith("T"):
        token = token.removeprefix("T")
        if token == "-1":
            trick = settings.num_tricks - 1
        else:
            assert token.isdigit(), token
            trick = int(token)
        return TrickCond(trick=trick, settings=settings, player=player)
    elif token.startswith("#T"):
        token = token.removeprefix("#T")
        direction, token = parse_dir(token)
        if token.isdigit():
            return CumTrickCond(
                direction=direction,
                num_tricks=int(token),
                settings=settings,
                player=player,
            )
        elif token.startswith("#T("):
            token = token.removeprefix("#T(").removesuffix(")")
            assert token in get_args(TrickCountType)
            return CumTrickCond(
                direction=direction,
                num_tricks=cast(TrickCountType, token),
                settings=settings,
                player=player,
            )
    elif token.startswith("#sweep"):
        token = token.removeprefix("#sweep>=")
        return SweepCond(num_sweeps=int(token), settings=settings, player=player)
    elif token.startswith("#"):
        token = token.removeprefix("#")
        card_filter, token = parse_card_filter(token)
        assert card_filter is not None
        direction, token = parse_dir(token)
        if token.startswith("#"):
            token = token.removeprefix("#")
            other_card_filter, token = parse_card_filter(token)
            assert other_card_filter is not None
            assert token == ""
            return CumCardCond(
                card_filter=card_filter,
                direction=direction,
                other_card_filter=other_card_filter,
                settings=settings,
                player=player,
            )
        else:
            assert token.isdigit()
            return CumCardCond(
                card_filter=card_filter,
                direction=direction,
                num_cards=int(token),
                settings=settings,
                player=player,
            )
    elif token.startswith("no("):
        token = token.removeprefix("no(").removesuffix(")")
        cond = parse_token(token, settings, player)
        assert isinstance(cond, (ConsecCond, TrickCond))
        cond.no = True
        return cond
    elif token.startswith("consec("):
        token = token.removeprefix("consec(").removesuffix(")")
        return ConsecCond(num_consec=int(token), settings=settings, player=player)
    elif token.startswith("nolead("):
        token = token.removeprefix("nolead(").removesuffix(")")
        return NoLeadCond(
            suits=[TO_SUIT_NUM[x] for x in token.split(",")],
            settings=settings,
            player=player,
        )
    elif token.startswith("with("):
        token = token.removeprefix("with(").removesuffix(")")
        card_filter, token = parse_card_filter(token)
        assert token == "" and card_filter is not None
        return WithCond(card_filter=card_filter, settings=settings, player=player)
    elif token.startswith("sum"):
        token = token.removeprefix("sum")
        direction, token = parse_dir(token)
        return SumCond(
            direction=direction, sum=int(token), settings=settings, player=player
        )
    else:
        card, token = parse_card(token)
        if card is not None:
            assert token == ""
            return CardCond(card=card, settings=settings, player=player)

    raise NotImplementedError("unhandled", orig_token)


class Task:
    def __init__(self, formula: str, desc: str = ""):
        self.formula = formula
        self.desc = desc


class AssignedTask(Task):
    def __init__(self, formula: str, player: int, settings: Settings, desc: str = ""):
        super().__init__(formula, desc=desc)
        self.player = player
        self.settings = settings
        self.parse_formula()
        self.status: Status = "unresolved"

    def parse_formula(self):
        tokens = self.formula.split()

        try:
            self.in_one_trick = False
            if tokens[0] == "1T":
                self.in_one_trick = True
                tokens = tokens[1:]

            self.conds = [
                parse_token(token, self.settings, self.player) for token in tokens
            ]
        except Exception as e:
            raise ValueError(f"Could not parse {self.formula}") from e

        for cond in self.conds:
            cond.reset()

    def on_trick_end(self, state: State) -> None:
        """on_trick_end task state with new state, on_trick_endd after every trick."""
        if self.status != "unresolved":
            return

        for cond in self.conds:
            if cond.status != "unresolved":
                continue
            cond.on_trick_end(state)
            if self.in_one_trick:
                cond.on_game_end()

        if all(cond.status == "success" for cond in self.conds):
            self.status = "success"

        if self.in_one_trick:
            for cond in self.conds:
                cond.reset()
                assert cond.status == "unresolved"

        if not self.in_one_trick and any(cond.status == "fail" for cond in self.conds):
            self.status = "fail"

    def on_game_end(self) -> None:
        """on_trick_end task state at end of game."""
        if self.status != "unresolved":
            return

        if self.in_one_trick:
            self.status = "fail"
            return

        for cond in self.conds:
            if cond.status != "unresolved":
                continue

            cond.on_game_end()

        assert all(cond.status != "unresolved" for cond in self.conds)
        if all(cond.status == "success" for cond in self.conds):
            self.status = "success"
        else:
            self.status = "fail"


TASK_DEFS = [
    ("1T 7p with(t)", "I will win 7p with a trump"),
    (
        "1T #rank(>6)=0 #t=0",
        "I will a trick of which the card values are greater than 6. No trumps are allowed in the trick.",
    ),
    (
        "1T #p=#b #p>0",
        "I will win as many pink as blue cards in one trick. 0 pink/blue cards is not allowed.",
    ),
    ("1T #8>=1 with(4)", "I will win an 8 with a 4."),
    ("1T with(5)", "I will win a trick using a 5"),
    ("1T with(3)", "I will win a trick using a 3"),
    (
        "1T sum>28 #t=0",
        "I will win a trick with a sum greater than 28. No trumps are allowed in the trick.",
    ),
    ("1T sum<12 #t=0", ""),
    ("1T sum>=22 sum<=23 #t=0", ""),
    ("1T #6>=2 with(6)", ""),
    ("1T #odd=0", ""),
    ("1T with(2)", ""),
    ("1T #rank(<=5)=0 #t=0", ""),
    ("1T #even=0", ""),
    ("1T T-1 2g", ""),
    ("1T with(6)", ""),
    ("1T #5>=1 with(7)", ""),
    ("1T 9g with(b)", ""),
    ("T0 T-1", ""),
    ("#T<#T(capt)", ""),
    ("T0 T1 T2", ""),
    ("6g", ""),
    ("#7>=2", ""),
    ("5p 6y", ""),
    ("#T=2", ""),
    ("#g=2", ""),
    ("#sweep>=1", ""),
    ("#T>#T(sumothers)", ""),
    ("#t=0", ""),
    ("#T=#T(capt)", ""),
    ("8p 5b", ""),
    ("#p>=5", ""),
    ("#t=2", ""),
    ("consec(2) #T=2", ""),
    ("#t=3", ""),
    ("T-1", ""),
    ("9p 8y", ""),
    ("3p", ""),
    ("9y 7b", ""),
    ("#T=1", ""),
    ("consec(3)", ""),
    ("#T=0", ""),
    ("no(T0) no(T1) no(T2)", ""),
    ("2t #t=1", ""),
    ("#p>=1 #g>=1 #y>=1 #b>=1", ""),
    ("#b=2", ""),
    ("#p=1", ""),
    ("#5=0", ""),
    ("3t", ""),
    ("#T>#T(capt)", ""),
    ("T0", ""),
    ("1y", ""),
    ("#T>#T(anyother)", ""),
    ("consec(3) #T=3", ""),
    ("#6=3", ""),
    ("T0 T1", ""),
    ("#t=1", ""),
    ("#p=0", ""),
    ("1t #t=1", ""),
    ("consec(2)", ""),
    ("1p 7g", ""),
    ("#9>=3", ""),
    ("T-1 #T=1", ""),
    ("1b 2b 3b", ""),
    ("#g=#y #g>0", ""),
    ("T0 #T=1", ""),
    ("#9=2", ""),
    ("nolead(y,p,b)", ""),
    ("#p>#g", ""),
    ("#y>=7", ""),
    ("#p=0 #b=0", ""),
    ("no(consec(2))", ""),
    ("#8=0 #9=0", ""),
    ("#1=0", ""),
    ("#p=1 #g=1", ""),
    ("3g 4y 5y", ""),
    ("no(T0) no(T1) no(T2) no(T3) no(T4)", ""),
    ("3b 3g 3y 3p", ""),
    ("#y>#b", ""),
    ("#g=0", ""),
    ("#y=0", ""),
    ("#1=0 #2=0 #3=0", ""),
    ("#T=4", ""),
    ("#5>=3", ""),
    ("#p=#y #p>0", ""),
    ("nolead(p,g)", ""),
    ("#T<#T(anyother)", ""),
    ("#9=0", ""),
    ("#y=0 #g=0", ""),
    ("no(T0) no(T1) no(T2) no(T3)", ""),
    ("6b 7y", ""),
    ("5g 8b", ""),
    ("9b 9p 9y 9g", ""),
    ("4b", ""),
]
