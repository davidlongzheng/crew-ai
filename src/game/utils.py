from typing import cast

import torch

from .settings import Settings
from .state import State
from .types import Action, Card


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


def encode_card(x: Card, settings: Settings) -> tuple[int, int]:
    return (x.rank - 1, settings.get_suit_idx(x.suit))


def encode_action(x: Action, settings: Settings) -> tuple[int, int]:
    if x.type in ["signal", "play"]:
        return encode_card(cast(Card, x.card), settings)
    elif x.type == "nosignal":
        return (settings.max_suit_length, settings.num_suits)
    elif x.type == "draft":
        return (cast(int, x.task_idx), -1)
    elif x.type == "nodraft":
        return (settings.num_task_defs, -1)
    else:
        raise ValueError(x.type)


def encode_hand(hand: list[Card], settings: Settings) -> list[tuple[int, int]]:
    return [encode_card(x, settings) for x in hand]


def encode_tasks(state: State) -> list[tuple[int, int]]:
    task_idxs = []
    for task_idx in state.unassigned_task_idxs:
        task_idxs.append((task_idx, state.num_players))

    for player, tasks in enumerate(state.assigned_tasks):
        player_idx = state.get_player_idx(player)
        task_idxs += [(task.task_idx, player_idx) for task in tasks]

    return task_idxs


def calc_trick_winner(active_cards: list[tuple[Card, int]]) -> int:
    lead_suit = active_cards[0][0].suit
    _, winner = max(
        active_cards,
        key=lambda x: (x[0].is_trump, x[0].suit == lead_suit, x[0].rank),
    )
    return winner
