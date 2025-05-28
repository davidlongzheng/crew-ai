import pytest
from pydantic import ValidationError

from game.types import Card


def test_card_creation() -> None:
    # Valid cards
    card1 = Card(rank=1, suit=0)
    assert str(card1) == "1b"

    card2 = Card(rank=4, suit=2)
    assert str(card2) == "4p"

    trump_card = Card(rank=3, suit=4)
    assert str(trump_card) == "3t"

    # Invalid cards
    with pytest.raises(ValidationError):
        Card(rank=0, suit=0)  # rank must be > 0
