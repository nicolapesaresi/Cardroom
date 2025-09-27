import pytest
from cardroom.briscola.game.cards import Card
from cardroom.briscola.game.player import BriscolaPlayer


def make_sample_hand():
    return [Card(1, 0), Card(2, 1), Card(3, 2)]


def test_player_initialization_and_reset():
    player = BriscolaPlayer(name="Joe Rizz")
    assert player.name == "Joe Rizz"
    assert player.points == 0
    assert player.hand == []
    assert player.taken_cards == []

    # after reset, state should clear
    player.hand = make_sample_hand()
    player.points = 50
    player.reset()
    assert player.points == 0
    assert player.hand == []
    assert player.taken_cards == []


def test_play_card_valid_removes_card():
    player = BriscolaPlayer("Arnoldo")
    player.hand = make_sample_hand()
    card = player.play_card(1)
    assert isinstance(card, Card)
    assert card.face_id == 2
    assert len(player.hand) == 2
    # remaining cards do not include the played one
    assert all(c.face_id != 2 for c in player.hand)


def test_play_card_invalid_type():
    player = BriscolaPlayer("Mighèl")
    player.hand = make_sample_hand()
    with pytest.raises(TypeError):
        player.play_card("not-an-int")


def test_play_card_index_out_of_range():
    player = BriscolaPlayer("Mallory")
    player.hand = make_sample_hand()
    # len(hand) == 3, so index 3 is invalid (0-based)
    with pytest.raises(IndexError):
        player.play_card(3)
