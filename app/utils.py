from app.types import Card


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
