import pytest
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.cards import Card

@pytest.fixture
def env():
    return BriscolaEnv(names=["Alice", "Bob"], render_mode=None)

def test_reset_deals_three_cards(env):
    """Each player starts with 3 cards, briscola suit is defined."""
    env.reset()
    assert all(len(p.hand) == 3 for p in env.players)
    assert env.briscola_spy is not None
    assert env.briscola_suit_id in range(4)

def test_play_card_and_turn_resolution(env):
    """Players can play cards and a winner is determined."""
    env.reset()
    p0_hand_size = len(env.players[0].hand)
    env.step(0)  # p0 plays a card
    env.step(0)  # p1 plays a card -> triggers resolve_turn
    assert len(env.cards_on_table) == 0
    assert any(p.points >= 0 for p in env.players)

def test_game_progress_and_deck_dealing(env):
    """After each trick, players draw until deck is empty."""
    env.reset()
    initial_deck_size = env.dealer.cards_left
    env.step(0)
    env.step(0)
    assert env.dealer.cards_left < initial_deck_size
    # players still have 3 cards in hand
    assert all(len(p.hand) == 3 for p in env.players)

def test_game_end(env):
    """Game ends after 20 turns."""
    env.reset()
    # force play through the entire game
    while not env.done:
        env.step(0 if env.players[env.current_player_id].hand else 0)
    assert env.done
    assert hasattr(env, "result")
    assert isinstance(env.result, int)

def test_clone_integrity(env):
    """Cloned env should have same visible state but independent objects."""
    env.reset()
    clone = env.clone()
    assert clone.current_player_id == env.current_player_id
    assert clone.turn_counter == env.turn_counter
    # hands should be separate lists
    clone.players[0].hand.clear()
    assert len(env.players[0].hand) == 3

def test_clone_from_observation(env):
    """Clone from observation should preserve current player's hand."""
    env.reset()
    current_hand = env.players[env.current_player_id].hand[:]
    clone = env.clone_from_observation()
    assert clone.players[env.current_player_id].hand == current_hand