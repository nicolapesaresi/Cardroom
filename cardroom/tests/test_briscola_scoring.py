import pytest
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.utils.scoring import play_game, play_n_games

def test_play_game_returns_types():
    """Test that play_game returns a string winner and a dict state."""
    agents = [RandomAgent(), RandomAgent()]
    winner, state = play_game(agents)
    
    assert isinstance(winner, str), "Winner should be a string"
    assert isinstance(state, dict), "State should be a dictionary"

def test_play_game_with_donatello():
    """Test that play_game works with a DonatelloAgent."""
    agents = [DonatelloAgent(name="MCTS"), RandomAgent()]
    winner, state = play_game(agents)
    
    assert isinstance(winner, str)
    assert isinstance(state, dict)

def test_play_n_games_returns_lists():
    """Test that play_n_games returns lists of winners and states."""
    agents = [RandomAgent(), RandomAgent()]
    winners, states = play_n_games(agents, n_games=5)
    
    assert isinstance(winners, list)
    assert isinstance(states, list)
    assert len(winners) == 5
    assert len(states) == 5
    for w in winners:
        assert isinstance(w, str)
    for s in states:
        assert isinstance(s, dict)

def test_play_n_games_edge_case_one_game():
    """Test play_n_games with n_games=1."""
    agents = [RandomAgent(), RandomAgent()]
    winners, states = play_n_games(agents, n_games=1)
    
    assert len(winners) == 1
    assert len(states) == 1

def test_play_n_games_invalid_n():
    """Test that negative n_games raises ValueError."""
    agents = [RandomAgent(), RandomAgent()]
    with pytest.raises(ValueError):
        play_n_games(agents, n_games=0)
    with pytest.raises(ValueError):
        play_n_games(agents, n_games=-5)

def test_play_game_winner_in_agents_or_draw():
    """Ensure that the winner returned by play_game is either one of the agents or 'Draw'."""
    agents = [RandomAgent(), RandomAgent()]
    winner, state = play_game(agents)

    agent_names = [agent.name for agent in agents] + ["Draw"]
    assert winner in agent_names, f"Winner {winner} not in {agent_names}"
    assert isinstance(state, dict), "State should be a dictionary"
    assert "player_names" in state, "Final state should contain player_names"
    # The winner should match the result if not a draw
    if winner != "Draw":
        winner_index = state["player_names"].index(winner)
        assert winner_index in [0, 1]

def test_play_n_games_winners_and_states_consistent():
    """Ensure that winners list and final states list have same length and valid entries."""
    agents = [RandomAgent(), DonatelloAgent(name="MCTS")]
    n_games = 5
    winners, states = play_n_games(agents, n_games)

    assert len(winners) == n_games, "Winners list length mismatch"
    assert len(states) == n_games, "States list length mismatch"

    agent_names = [agent.name for agent in agents] + ["Draw"]
    for winner, state in zip(winners, states):
        assert winner in agent_names, f"Winner {winner} not in {agent_names}"
        assert isinstance(state, dict)
        assert "player_names" in state, "Final state should contain player_names"
        # Optional: check winner matches result in final state
        if winner != "Draw":
            winner_index = state["player_names"].index(winner)
            assert winner_index in [0, 1]