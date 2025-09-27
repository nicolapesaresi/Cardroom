import pytest
import numpy as np

from cardroom.briscola.game.cards import Card, FACES, SUITS
from cardroom.briscola.game.dealer import BriscolaDealer

def test_initialize_deck_creates_full_deck():
    dealer = BriscolaDealer(seed=42)
    assert dealer.cards_left == len(FACES) * len(SUITS)
    # Each combination of face + suit is present
    seen = {(c.face_id, c.suit_id) for c in dealer.deck}
    assert seen == {(f, s) for f in FACES for s in SUITS}


def test_shuffle_changes_order():
    dealer1 = BriscolaDealer(seed=123)
    dealer2 = BriscolaDealer(seed=123)  # same seed → same shuffle
    dealer3 = BriscolaDealer(seed=999)  # different seed → different shuffle

    order1 = [(c.face_id, c.suit_id) for c in dealer1.deck]
    order2 = [(c.face_id, c.suit_id) for c in dealer2.deck]
    order3 = [(c.face_id, c.suit_id) for c in dealer3.deck]

    assert order1 == order2
    assert order1 != order3


def test_set_briscola_marks_correct_suit():
    dealer = BriscolaDealer(seed=0)
    briscola_suit = dealer.spy.suit_id
    for card in dealer.deck:
        if card.suit_id == briscola_suit:
            assert card.rank >= 100
        else:
            assert card.rank < 100


def test_get_cards_left_and_deal():
    dealer = BriscolaDealer(seed=1)
    initial_count = dealer.get_cards_left()
    card = dealer.deal()
    assert isinstance(card, Card)
    assert dealer.get_cards_left() == initial_count - 1


def test_deal_until_empty_then_raises():
    dealer = BriscolaDealer(seed=2)
    while dealer.get_cards_left() > 0:
        dealer.deal()
    assert dealer.get_cards_left() == 0
    with pytest.raises(IndexError):
        dealer.deal()


def test_draw_starting_player_within_range():
    n_players = 4
    for _ in range(50):
        idx = BriscolaDealer.draw_starting_player(n_players)
        assert 0 <= idx < n_players


def test_clone_creates_independent_copy():
    dealer = BriscolaDealer(seed=3)
    clone = dealer.clone()

    # Different object
    assert clone is not dealer
    # Decks have same structure but are different objects
    assert [(c.face_id, c.suit_id) for c in dealer.deck] == [
        (c.face_id, c.suit_id) for c in clone.deck
    ]
    assert all(orig is not copy for orig, copy in zip(dealer.deck, clone.deck))

    # spy should be cloned as well
    assert dealer.spy.face_id == clone.spy.face_id
    assert dealer.spy is not clone.spy
