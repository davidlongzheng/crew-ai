import torch

from .settings import Settings
from .types import Card


def split_by_suit(hand: list[Card]) -> list[list[Card]]:
    hand = sorted(hand, key=lambda x: (x.suit, x.rank))
    ret: list[list[Card]] = []
    prev_suit: int | None = None
    for card in hand:
        if card.suit != prev_suit:
            ret.append([])

        ret[-1].append(card)
        prev_suit = card.suit

    return ret


def to_card(arr: torch.Tensor, settings) -> Card:
    rank = int(arr[0].item()) + 1
    suit = settings.get_suit(int(arr[1].item()))
    return Card(rank, suit)


def to_hand(arr: torch.Tensor, settings) -> list[Card]:
    return [to_card(x, settings) for x in arr if x[0].item() != -1]


def card_to_tuple(x: Card | None, settings: Settings) -> tuple[int, int]:
    if x is None:
        return (settings.max_suit_length, settings.num_suits)
    return (x.rank - 1, settings.get_suit_idx(x.suit))


def encode_hand(hand: list[Card], settings: Settings) -> list[tuple[int, int]]:
    return [card_to_tuple(x, settings) for x in hand]


def calc_trick_winner(active_cards: list[tuple[Card, int]]) -> int:
    lead_suit = active_cards[0][0].suit
    _, winner = max(
        active_cards,
        key=lambda x: (x[0].is_trump, x[0].suit == lead_suit, x[0].rank),
    )
    return winner
