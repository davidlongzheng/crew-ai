from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, cast, get_args, override

from .settings import Settings
from .types import TO_SUIT_NUM, Card

if TYPE_CHECKING:
    from .state import State

Status = Literal["success", "fail", "unresolved"]
Direction = Literal[">", "<", ">=", "=", "<="]
CardFilter = Callable[[Card], bool]
TrickCountType = Literal["capt", "sumothers", "anyother"]


def clamp(x):
    return max(min(x, 1), 0)


@dataclass(kw_only=True)
class Condition:
    settings: Settings
    player: int
    # Once status changes from unresolved, cannot change again.
    # Status should change as soon as can be resolved.
    status: Status = "unresolved"
    # Should always be a number between 0 and 1.
    # Does not need to be monotonically non-decreasing.
    partial_value: float = 0.0

    def reset(self):
        pass

    def on_trick_end(self, state: State):
        pass

    def on_end(self):
        pass


@dataclass
class TrickCond(Condition):
    trick: int
    no: bool = False

    @override
    def reset(self):
        self.status = "unresolved"

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick == self.trick:
            won_trick = state.trick_winner == self.player
            self.status = "success" if won_trick == (not self.no) else "fail"

    @override
    def on_end(self):
        assert self.status != "unresolved"


@dataclass
class CardCond(Condition):
    card: Card

    @override
    def reset(self):
        self.status = "unresolved"

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if any(self.card == card for card, _ in state.active_cards):
            self.status = "success" if state.trick_winner == self.player else "fail"

    @override
    def on_end(self):
        if self.status == "unresolved":
            self.status = "fail"


@dataclass
class CumTrickCond(Condition):
    direction: Direction
    num_tricks: int | TrickCountType

    num_tricks_won: int = 0
    num_other_tricks_won: int = 0
    num_other_tricks_won_per_player: list[int] = field(default_factory=list)

    @override
    def reset(self):
        self.status = "unresolved"
        self.partial_value = 0
        self.num_tricks_won = 0
        if self.num_tricks == "anyother":
            self.num_other_tricks_won_per_player = [
                0 for _ in range(self.settings.num_players)
            ]
        else:
            self.num_other_tricks_won = 0

    def get_num_other_tricks(self):
        if isinstance(self.num_tricks, int):
            return self.num_tricks
        elif self.num_tricks in {"capt", "sumothers"}:
            return self.num_other_tricks_won
        else:
            assert self.num_tricks == "anyother"
            num_other_tricks_won_per_player = [
                x
                for i, x in enumerate(self.num_other_tricks_won_per_player)
                if i != self.player
            ]
            return (
                max(num_other_tricks_won_per_player)
                if self.direction in [">=", ">"]
                else min(num_other_tricks_won_per_player)
            )

    @override
    def on_trick_end(self, state: State):
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

        num_other_tricks = self.get_num_other_tricks()
        comp_ok = compare_dir(self.num_tricks_won, num_other_tricks, self.direction)

        if isinstance(self.num_tricks, int):
            if self.direction in [">=", ">"] and comp_ok:
                self.status = "success"
            elif self.direction in ["<", "<="] and not comp_ok:
                self.status = "fail"
            elif self.direction == "=" and self.num_tricks_won > num_other_tricks:
                self.status = "fail"

        if self.direction in [">=", ">"]:
            self.partial_value = clamp(1 - (num_other_tricks - self.num_tricks_won) / 3)
        elif self.direction in ["<=", "<"]:
            self.partial_value = clamp(1 - (self.num_tricks_won - num_other_tricks) / 3)
        else:
            self.partial_value = clamp(
                1 - abs(self.num_tricks_won - num_other_tricks) / 3
            )

    @override
    def on_end(self):
        if self.status == "unresolved":
            self.status = (
                "success"
                if compare_dir(
                    self.num_tricks_won, self.get_num_other_tricks(), self.direction
                )
                else "fail"
            )


@dataclass
class CumCardCond(Condition):
    direction: Direction
    card_filter: CardFilter
    num_cards: int | None = None
    other_card_filter: CardFilter | None = None

    # Track cards won that match each filter
    num_cards_won: int = 0
    num_other_cards_won: int = 0

    @override
    def reset(self):
        self.status = "unresolved"
        self.partial_value = 0
        self.num_cards_won = 0
        self.num_other_cards_won = 0

    def get_num_other_cards(self):
        return (
            self.num_cards if self.num_cards is not None else self.num_other_cards_won
        )

    @override
    def on_trick_end(self, state: State):
        if state.trick_winner == self.player:
            # Count cards in this trick that match our filters
            self.num_cards_won += sum(
                1 for card, _ in state.active_cards if self.card_filter(card)
            )
            if self.other_card_filter:
                self.num_other_cards_won += sum(
                    1 for card, _ in state.active_cards if self.other_card_filter(card)
                )

        num_other_cards = self.get_num_other_cards()
        comp_ok = compare_dir(self.num_cards_won, num_other_cards, self.direction)

        if self.num_cards is not None:
            if self.direction in [">=", ">"] and comp_ok:
                self.status = "success"
            elif self.direction in ["<", "<="] and not comp_ok:
                self.status = "fail"
            elif self.direction == "=" and self.num_cards > num_other_cards:
                self.status = "fail"

        if self.direction in [">=", ">"]:
            self.partial_value = clamp(1 - (num_other_cards - self.num_cards_won) / 3)
        elif self.direction in ["<=", "<"]:
            self.partial_value = clamp(1 - (self.num_cards_won - num_other_cards) / 3)
        else:
            self.partial_value = clamp(
                1 - abs(self.num_cards_won - num_other_cards) / 3
            )

    @override
    def on_end(self):
        if self.status == "unresolved":
            self.status = (
                "success"
                if compare_dir(
                    self.num_cards_won, self.get_num_other_cards(), self.direction
                )
                else "fail"
            )


@dataclass
class WithCond(Condition):
    card_filter: CardFilter

    @override
    def reset(self):
        self.status = "unresolved"

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            (player_card,) = [
                card for card, player in state.active_cards if player == self.player
            ]
            if self.card_filter(player_card):
                self.status = "success"

    @override
    def on_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class SweepCond(Condition):
    num_sweeps: int

    # Track cards won per suit
    cards_won_per_suit: dict[int, int] = field(default_factory=dict)

    @override
    def reset(self):
        self.status = "unresolved"
        self.partial_value = 0
        self.cards_won_per_suit = {i: 0 for i in range(self.settings.num_side_suits)}

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            # Add all cards from this trick to their respective suit sets
            for card, _ in state.active_cards:
                if card.is_trump:
                    continue
                self.cards_won_per_suit[card.suit] += 1

            num_sweeps = sum(
                x == self.settings.side_suit_length
                for x in self.cards_won_per_suit.values()
            )
            self.partial_value = num_sweeps / self.num_sweeps
            if num_sweeps >= self.num_sweeps:
                self.status = "success"

    @override
    def on_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class ConsecCond(Condition):
    num_consec: int
    no: bool = False

    cur_consec_start: int | None = None
    cur_consec_end: int | None = None

    def __post_init__(self):
        assert self.num_consec >= 2

    @override
    def reset(self):
        self.status = "unresolved"
        self.partial_value = 0
        self.cur_consec_start = None
        self.cur_consec_end = None

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            if self.cur_consec_end and state.trick == self.cur_consec_end + 1:
                self.cur_consec_end += 1
            else:
                self.cur_consec_start = state.trick
                self.cur_consec_end = state.trick
        else:
            self.cur_consec_start = None
            self.cur_consec_end = None

        if self.no:
            if (
                self.cur_consec_start is not None
                and self.cur_consec_end is not None
                and (self.cur_consec_end - self.cur_consec_start + 1) >= self.num_consec
            ):
                self.status = "fail"
            else:
                self.partial_value = (state.trick + 1) / self.settings.num_tricks
        else:
            if self.cur_consec_start is not None and self.cur_consec_end is not None:
                num_consec = self.cur_consec_end - self.cur_consec_start + 1
                self.partial_value = max(
                    self.partial_value, num_consec / self.num_consec
                )
                if num_consec >= self.num_consec:
                    self.status = "success"

    @override
    def on_end(self):
        if self.status == "unresolved":
            self.status = "success" if self.no else "fail"


@dataclass
class SumCond(Condition):
    direction: Direction
    sum: int

    @override
    def reset(self):
        self.status = "unresolved"

    @override
    def on_trick_end(self, state: State):
        if self.status != "unresolved":
            return

        if state.trick_winner == self.player:
            trick_sum = sum(card.rank for card, _ in state.active_cards)
            if compare_dir(trick_sum, self.sum, self.direction):
                self.status = "success"

    @override
    def on_end(self):
        if self.status != "success":
            self.status = "fail"


@dataclass
class NoLeadCond(Condition):
    suits: list[int]

    @override
    def reset(self):
        self.status = "unresolved"
        self.partial_value = 0
        self.num_bad_lead = 0

    @override
    def on_trick_end(self, state: State):
        # If we're the leader and we lead a forbidden suit, fail
        if state.leader == self.player and state.active_cards[0][0].suit in self.suits:
            self.status = "fail"
            self.num_bad_lead += 1

        self.partial_value = clamp(1 - self.num_bad_lead / 2)

    @override
    def on_end(self):
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
    def __init__(self, formula: str, desc: str, difficulty: int, task_idx: int):
        self.formula = formula
        self.desc = desc or formula
        self.difficulty = difficulty
        self.task_idx = task_idx

    def __eq__(self, other):
        return isinstance(other, Task) and self.formula == other.formula

    def __hash__(self):
        return hash(self.formula)

    def __str__(self):
        return self.desc

    def __repr__(self):
        return f"Task({self.desc})"


class AssignedTask(Task):
    def __init__(
        self,
        formula: str,
        desc: str,
        difficulty: int,
        task_idx: int,
        player: int,
        settings: Settings,
    ):
        super().__init__(formula, desc, difficulty, task_idx)
        self.player = player
        self.settings = settings
        self.parse_formula()
        self.status: Status = "unresolved"
        self.value = 0

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

    def compute_value(self):
        if self.in_one_trick:
            self.value = (
                1 if self.status == "success" else -1 if self.status == "fail" else 0
            )
        else:
            cond_values = []
            for cond in self.conds:
                assert 0 <= cond.partial_value <= 1
                partial_value = (
                    1 if cond.status == "success" else cond.partial_value * 2 - 1
                )
                cond_values.append(partial_value)
            avg_cond_value = sum(cond_values) / len(cond_values)
            assert -1 <= avg_cond_value <= 1
            task_bonus = (
                self.settings.task_bonus
                if self.status == "success"
                else -self.settings.task_bonus
                if self.status == "fail"
                else 0
            )
            self.value = (avg_cond_value + task_bonus) / (self.settings.task_bonus + 1)

        assert -1 <= self.value <= 1

    def on_trick_end(self, state: State) -> None:
        """on_trick_end task state with new state, on_trick_end after every trick."""
        if self.in_one_trick and self.status != "unresolved":
            return

        for cond in self.conds:
            cond.on_trick_end(state)
            if self.in_one_trick:
                cond.on_end()

        if all(cond.status == "success" for cond in self.conds):
            assert self.status in ["unresolved", "success"]
            self.status = "success"

        if self.in_one_trick:
            for cond in self.conds:
                cond.reset()
                assert cond.status == "unresolved"

        if not self.in_one_trick and any(cond.status == "fail" for cond in self.conds):
            assert self.status in ["unresolved", "fail"]
            self.status = "fail"

        self.compute_value()

    def on_game_end(self) -> None:
        """on_trick_end task state at end of game."""
        if self.in_one_trick:
            self.status = "fail"
            self.compute_value()
            return

        for cond in self.conds:
            cond.on_end()

        assert all(cond.status != "unresolved" for cond in self.conds)
        if all(cond.status == "success" for cond in self.conds):
            assert self.status in ["unresolved", "success"]
            self.status = "success"
        else:
            assert self.status in ["unresolved", "fail"]
            self.status = "fail"

        self.compute_value()


TASK_DEFS = [
    ("1T 7p with(t)", "I will win 7p with a submarine.", 3),
    (
        "1T #rank(>6)=0 #t=0",
        "I will win a trick of which the card values are all less than 7. Submarines are not allowed in the trick.",
        3,
    ),
    (
        "1T #p=#b #p>0",
        "I will win as many pink as blue cards in one trick. 0 pink/blue cards is not allowed.",
        3,
    ),
    ("1T #8>=1 with(4)", "I will win an 8 with a 4.", 4),
    ("1T with(5)", "I will win a trick using a 5", 3),
    ("1T with(3)", "I will win a trick using a 3", 4),
    (
        "1T sum>28 #t=0",
        "I will win a trick with a total value greater than 28. Submarines are not allowed in the trick.",
        3,
    ),
    (
        "1T sum<12 #t=0",
        "I will win a trick with a total value less than 12. Submarines are not allowed in the trick.",
        3,
    ),
    (
        "1T sum>=22 sum<=23 #t=0",
        "I will win a trick with a total value of 22 or 23. Submarines are not allowed in the trick.",
        3,
    ),
    ("1T #6>=2 with(6)", "I will win a 6 with another 6.", 3),
    ("1T #odd=0", "I will win a trick that contains only even-numbered cards.", 5),
    (
        "1T #g=#y #g>0",
        "I will win as many green as yellow cards in one trick. 0 green/yellow cards is not allowed.",
        3,
    ),
    ("1T with(2)", "I will win a trick using a 2.", 4),
    (
        "1T #rank(<=5)=0",
        "I will win a trick of which the card values are all greater than 5.",
        4,
    ),
    ("1T #even=0", "I will win a trick that contains only odd-numbered cards.", 4),
    ("1T T-1 2g", "I will win 2g in the final trick of the game.", 4),
    ("1T with(6)", "I will win a trick using a 6.", 3),
    ("1T #5>=1 with(7)", "I will win a 5 with a 7.", 2),
    ("1T 9g with(t)", "I will the 9g with a submarine.", 3),
    ("T0 T-1", "I will win the frist and the last trick.", 4),
    (
        "#T<#T(capt)",
        "I will win fewer tricks than the captain. I am not the captain",
        2,
    ),
    ("T0 T1 T2", "I will win the first 3 tricks.", 3),
    ("6g", "I will win 6g", 1),
    ("#7>=2", "I will win at least 2 7's.", 2),
    ("5p 6y", "I will win 5p 6y.", 2),
    ("#T=2", "I will win exactly 2 tricks.", 2),
    ("#g=2", "I will win exactly 2 greens.", 4),
    ("#sweep>=1", "I will win all the cards in at least one of the 4 colors.", 4),
    ("#T>#T(sumothers)", "I will more tricks than everyone else combined.", 4),
    ("#t=0", "I will win no submarines.", 1),
    (
        "#T=#T(capt)",
        "I will win as many tricks as the captain. I am not the captain.",
        3,
    ),
    ("8p 5b", "I will win 8p and 5b.", 2),
    ("#p>=5", "I will at least 5 pinks.", 3),
    ("#t=2", "I will win exactly 2 submarines.", 3),
    ("consec(2) #T=2", "I will win exactly 2 tricks and they will be in a row.", 3),
    ("#t=3", "I will win exactly 3 submarines.", 4),
    ("T-1", "I will win the last trick.", 3),
    ("9p 8y", "I will win 9p 8y.", 3),
    ("3p", "I will win 3p.", 1),
    ("9y 7b", "I will win 9y and 7b.", 3),
    ("#T=1", "I will win exactly 1 trick.", 2),
    ("consec(3)", "I will win 3 tricks in a row.", 3),
    ("#T=0", "I will 0 tricks.", 3),
    ("no(T0) no(T1) no(T2)", "I will win none of the first 3 tricks.", 2),
    ("2t #t=1", "I will 2t and no other submarine.", 3),
    ("#p>=1 #g>=1 #y>=1 #b>=1", "I will win at least one card of each color.", 3),
    ("#b=2", "I will win exactly 2 blues.", 4),
    ("#p=1", "I will win exactly 1 pink.", 3),
    ("#5=0", "I will no 5", 2),
    ("3t", "I will win 3t.", 1),
    (
        "#T>#T(capt)",
        "I will win more tricks than the captain. I am not the captain.",
        2,
    ),
    ("T0", "I will win the first trick.", 1),
    ("1y", "I will win 1y.", 1),
    ("#T>#T(anyother)", "I will win more tricks than anyone else.", 3),
    ("consec(3) #T=3", "I will win exactly 3 tricks and they will be in a row.", 3),
    ("#6=3", "I will win exactly 3 6's.", 4),
    ("T0 T1", "I will win the first 2 tricks.", 1),
    ("#t=1", "I will win exactly 1 submarine.", 3),
    ("#p=0", "I will win no pink.", 2),
    ("1t #t=1", "I will win 1t and no other submarine.", 3),
    ("consec(2)", "I will win 2 tricks in a row.", 1),
    ("1p 7g", "I will win 1p and 7g.", 2),
    ("#9>=3", "I will win at least 3 9's.", 4),
    ("T-1 #T=1", "I will win only the last trick.", 4),
    ("1b 2b 3b", "I will win 1b 2b 3b.", 3),
    ("T0 #T=1", "I will win only the first trick.", 3),
    ("#9=2", "I will win exactly 2 9's.", 3),
    ("nolead(y,p,b)", "I will not open a trick with yellow, pink, or blue.", 3),
    ("#p>#g", "I will win more pink than green cards. 0 green cards is allowed.", 1),
    ("#y>=7", "I will win at least 7 yellows.", 3),
    ("#p=0 #b=0", "I will win no pink or blues.", 3),
    ("no(consec(2))", "I will never win 2 tricks in a row.", 2),
    ("#8=0 #9=0", "I will win no 8 or 9's.", 3),
    ("#1=0", "I will win no 1.", 2),
    ("#p=1 #g=1", "I will win exactly 1p and 1g.", 4),
    ("3g 4y 5y", "I will 3g 4y and 5y.", 4),
    ("no(T0) no(T1) no(T2) no(T3) no(T4)", "I will none of the first 5 tricks.", 3),
    ("3b 3g 3y 3p", "I will win 3b 3g 3y 3p.", 4),
    ("#y>#b", "I will more yellow than blue cards. 0 blue cards is allowed", 1),
    ("#g=0", "I will win no greens.", 2),
    ("#y=0", "I will no yellows.", 2),
    ("#1=0 #2=0 #3=0", "I will no win 1, 2, or 3's.", 3),
    ("#T=4", "I will win exactly 4 tricks.", 3),
    ("#5>=3", "I will win at least 3 5's.", 4),
    (
        "#p=#y #p>0",
        "I will win as many pink as yellow cards. 0 pink/yellow cards is not allowed.",
        4,
    ),
    ("nolead(p,g)", "I will not open with a pink or green.", 1),
    ("#T<#T(anyother)", "I will win fewer tricks than anyone else.", 2),
    ("#9=0", "I will win no 9.", 1),
    ("#y=0 #g=0", "I will win no yellow or greens.", 3),
    ("no(T0) no(T1) no(T2) no(T3)", "I will win none of the first 4 tricks.", 2),
    ("6b 7y", "I will the 6b and 7y.", 2),
    ("5g 8b", "I will win the 5g and 8b.", 2),
    ("9b 9p 9y 9g", "I will win 9b 9p 9y 9g.", 5),
    ("4b", "I will win 4b.", 1),
]

EASY_TASK_DEFS = [
    ("#T>=1", "I will win at least one trick.", 1),
    ("#T>=2", "I will win at least two tricks.", 1),
]

MED_TASK_DEFS = [
    ("4b", "I will win 4b.", 1),
    ("nolead(p,g)", "I will not open with a pink or green.", 1),
    ("3p", "I will win 3p.", 1),
    ("1y", "I will win 1y.", 1),
    ("#t=0", "I will win no submarines.", 1),
    ("3t", "I will win 3t.", 1),
    ("T0", "I will win the first trick.", 1),
    ("6g", "I will win 6g", 1),
    ("#p>#g", "I will win more pink than green cards. 0 green cards is allowed.", 1),
    ("T0 T1", "I will win the first 2 tricks.", 1),
    ("#T<#T(anyother)", "I will win fewer tricks than anyone else.", 2),
    ("#y=0", "I will no yellows.", 2),
    ("#T=2", "I will win exactly 2 tricks.", 2),
    ("#1=0", "I will win no 1.", 2),
    ("#p=0", "I will win no pink.", 2),
    ("consec(2) #T=2", "I will win exactly 2 tricks and they will be in a row.", 3),
    ("9y 7b", "I will win 9y and 7b.", 3),
    ("consec(3) #T=3", "I will win exactly 3 tricks and they will be in a row.", 3),
    ("1T 7p with(t)", "I will win 7p with a submarine.", 3),
    ("1T with(6)", "I will win a trick using a 6.", 3),
]


def get_task_defs(bank):
    if bank == "all":
        return TASK_DEFS
    elif bank == "easy":
        return EASY_TASK_DEFS
    elif bank == "med":
        return MED_TASK_DEFS
