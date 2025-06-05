from typing import cast

import torch

import cpp_game
from game.settings import Settings
from game.state import State
from game.types import Action, Card


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


def to_action(arr: torch.Tensor, phase_idx: int, player: int, settings) -> Action:
    phase = settings.get_phase(phase_idx)
    if phase == "draft":
        if arr[0].item() == settings.num_task_defs:
            return Action(player, "nodraft")
        else:
            return Action(player, "draft", task_idx=int(arr[0].item()))
    elif phase == "signal":
        if arr[0].item() == settings.max_suit_length:
            return Action(player, "nosignal")
        else:
            return Action(player, "signal", card=to_card(arr, settings))
    elif phase == "play":
        return Action(player, "play", card=to_card(arr, settings))
    else:
        raise ValueError(phase)


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


def get_splits_and_phases(settings, as_phase_index=True):
    if settings.use_drafting:
        splits = [settings.num_players * settings.num_draft_tricks]
        phases = ["draft"]
    else:
        splits = []
        phases = []

    if settings.use_signals:
        if settings.single_signal:
            splits += [
                settings.num_players,
                settings.num_players * settings.num_tricks,
            ]
            phases += ["signal", "play"]
        else:
            splits += [settings.num_players] * (settings.num_tricks * 2)
            phases += ["signal", "play"]
    else:
        splits += [settings.num_players * settings.num_tricks]
        phases += ["play"]

    assert sum(splits) == settings.get_seq_length()
    assert len(splits) == len(phases)

    if as_phase_index:
        phases = [settings.get_phase_idx(x) for x in phases]
        assert all(0 <= x < settings.num_phases for x in phases)

    return splits, phases


def to_cpp_card(card):
    return cpp_game.Card(card.rank, card.suit)


def to_cpp_action(action):
    return cpp_game.Action(
        action.player,
        to_cpp_action_type(action.type),
        to_cpp_card(action.card) if action.card else None,
        action.task_idx,
    )


def to_cpp_action_type(action_type):
    return getattr(cpp_game.ActionType, action_type)


def to_py_card(cpp_card):
    return Card(rank=cpp_card.rank, suit=cpp_card.suit)


def to_py_action(cpp_action):
    return Action(
        player=cpp_action.player,
        type=to_py_action_type(cpp_action.type),
        card=to_py_card(cpp_action.card) if cpp_action.card else None,
        task_idx=cpp_action.task_idx,
    )


def to_py_action_type(cpp_action_type):
    return cpp_action_type.name
