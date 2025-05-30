from __future__ import annotations

import copy
import random

from game.state import State
from game.types import Card


def sample_determinization(state: State) -> State:
    # Does not work with drafting or signals yet.
    # For signals, we have to narrow our valid determinizations
    # even more.
    assert state.phase == "play"

    cards_unseen = [
        card
        for player, hand in enumerate(state.hands)
        if player != state.cur_player
        for card in hand
    ]
    random.shuffle(cards_unseen)
    player_map = [
        player
        for player, hand in enumerate(state.hands)
        if player != state.cur_player
        for _ in hand
    ]

    def is_valid(card, i):
        return not state.shown_out[player_map[i]][card.suit]

    for i, card in enumerate(cards_unseen):
        if is_valid(card, i):
            continue

        js = [
            j
            for j, card2 in enumerate(cards_unseen)
            if j != i and is_valid(card, j) and is_valid(card2, i)
        ]
        assert js

        j = random.choice(js)
        cards_unseen[i], cards_unseen[j] = cards_unseen[j], cards_unseen[i]

    new_hands: list[list[Card]] = []
    i = 0
    for player, hand in enumerate(state.hands):
        if player == state.cur_player:
            new_hands.append(hand.copy())
        else:
            new_hands.append(cards_unseen[i : i + len(hand)])
            i += len(hand)

    new_state = copy.deepcopy(state)
    new_state.hands = new_hands
    return new_state
