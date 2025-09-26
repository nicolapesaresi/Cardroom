from tqdm import tqdm

from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.donatello import DonatelloAgent

def play_game(agents: list, render_mode: str | None = None) -> tuple[str, dict]:
    """Plays one game of Briscola.
    Args:
        agents: list of agents to play the game.
        render_mode: rendering mode of the game, default is None.
    Returns:
        winner: string name of the winner (or 'Draw').
        final_state: final state of the game.
    """    
    names = [agent.name for agent in agents]
    env = BriscolaEnv(names, render_mode=render_mode)
    # check valid agents
    for agent in agents:
        if not isinstance(agent, Agent):
            raise NotImplementedError(f"Expected class Agent, got {type(agent)}")
        if isinstance(agent, HumanAgent) and render_mode == "pygame":
            agent.set_pygame_action_retriever(env.pygame)

    while not env.done:
        player_id = env.current_player_id
        agent = agents[player_id]
        if isinstance(agent, DonatelloAgent): # ugly. should be the same call for all agents
            action = agent.select_action(env.get_observation(), env.clone_from_observation())
        else:
            action = agent.select_action(env.get_observation())
        env.step(action)
    
    result = env.result
    final_state = env.get_state()
    winner_id = 0 if result == 1 else 1
    winner_id = None if result == 0 else winner_id # draws
    winner = final_state["player_names"][result] if winner_id is not None else "Draw"

    return winner, final_state

def play_n_games(agents: list, n_games: int, render_mode: str|None = None) -> tuple[list[str], list[dict]]: 
    """Plays n games of Briscola.
    Args:
        agents: list of agents to play the games.
        n_games: number of games to play.
        render_mode: rendering mode of the games, default is None.
    Returns:
        winners: list of string names of the winner (or 'Draw').
        final_states: list of final states of the game.
    """    
    if n_games < 1:
        raise ValueError("Cannot play negative games.")
    
    winners = []
    final_states = []
    for i in tqdm(range(n_games)):
        winner, state = play_game(agents, render_mode)
        winners.append(winner)
        final_states.append(state)

        # partial summary
        if i % (n_games // 10) == 0:
            print("Partial Summary:")
            for agent in agents:
                print(f"{agent.name} wins: {winners.count(agent.name)}")
            print(f"Draws: {winners.count("Draw")}")
    
    # summary
    print("Final Summary:")
    for agent in agents:
        print(f"{agent.name} wins: {winners.count(agent.name)}")
    print(f"Draws: {winners.count("Draw")}")
    
    return winners, final_states