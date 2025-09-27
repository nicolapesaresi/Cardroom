import pytest
from unittest.mock import patch, MagicMock

from cardroom.briscola.game.cards import Card, CardRetro, FACES, SUITS  # adjust import path


def test_card_initialization_valid():
    card = Card(1, 0)  # Ace of Ori
    assert card.face == FACES[1]
    assert card.suit == SUITS[0]
    assert card.points == 11
    assert card.rank == 10
    assert str(card) == "A of Ori"


def test_card_invalid_face():
    with pytest.raises(KeyError):
        Card(99, 0)


def test_card_invalid_suit():
    with pytest.raises(KeyError):
        Card(1, 99)


def test_card_equality():
    c1 = Card(1, 0)
    c2 = Card(1, 0)
    c3 = Card(2, 0)
    assert c1 == c2
    assert c1 != c3
    assert c1 != "not a card"