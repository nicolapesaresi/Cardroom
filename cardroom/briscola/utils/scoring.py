from tqdm import tqdm

from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.superbot import SuperbotAgent
from cardroom.briscola.agents.optimus import OptimusAgent
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
        elif isinstance(agent, SuperbotAgent) or isinstance(agent, OptimusAgent):
            action = agent.select_action(env.get_observation(), env.clone())
        else:
            action = agent.select_action(env.get_observation())
        env.step(action)
    
    result = env.result
    final_state = env.get_state()
    if result == 1:
        winner = final_state["player_names"][0]
    elif result == -1:
        winner = final_state["player_names"][1]
    else:
        winner = "Draw"


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
    with tqdm(total=n_games, desc="Playing games...") as pbar:
        for i in range(n_games):
            winner, state = play_game(agents, render_mode)
            winners.append(winner)
            final_states.append(state)

            # tqdm bar
            desc_str = " | ".join([f"{agent.name} wins: {winners.count(agent.name)}" for agent in agents])
            desc_str = desc_str + f" | Draws: {winners.count("Draws")}"
            pbar.set_description(f"Playing games... ({desc_str})")
            pbar.update(1)

    # summary
    print("Final Summary:")
    for agent in agents:
        print(f"{agent.name} wins: {winners.count(agent.name)}")
    print(f"Draws: {winners.count("Draw")}")
    
    return winners, final_states

def play_flipped_games(base_agents: list, render_mode: str | None = None) -> tuple[list[str], list[dict]]:
    """Plays two games of Briscola: one standard, and one where the env is flipped, meaning the two players play with opposite cards and same deck.
    Args:
        base_agents: list of agents to play the game (not flipped).
        render_mode: rendering mode of the game, default is None.
    Returns:
        winners: string name of the winner (or 'Draw').
        final_states: final state of the game.
    """
    winners = []
    final_states = []

    names = [agent.name for agent in base_agents]
    base_env = BriscolaEnv(names, render_mode=render_mode)

    # check valid agents
    for agent in base_agents:
        if not isinstance(agent, Agent):
            raise NotImplementedError(f"Expected class Agent, got {type(agent)}")
        if isinstance(agent, HumanAgent) and render_mode == "pygame":
            agent.set_pygame_action_retriever(base_env.pygame)

    flipped_env = base_env.clone()
    # switch players
    flipped_env.players[0].name = names[1]
    flipped_env.players[1].name = names[0]
    flipped_agents = [base_agents[1], base_agents[0]]

    for env, agents in zip([base_env, flipped_env], [base_agents, flipped_agents]):
        while not env.done:
            player_id = env.current_player_id
            agent = agents[player_id]
            if isinstance(agent, DonatelloAgent): # ugly. should be the same call for all agents
                action = agent.select_action(env.get_observation(), env.clone_from_observation())
            elif isinstance(agent, SuperbotAgent) or isinstance(agent, OptimusAgent):
                action = agent.select_action(env.get_observation(), env.clone())
            else:
                action = agent.select_action(env.get_observation())
            env.step(action)
        
        result = env.result
        final_state = env.get_state()
        if result == 1:
            winner = final_state["player_names"][0]
        elif result == -1:
            winner = final_state["player_names"][1]
        else:
            winner = "Draw"
        winners.append(winner)
        final_states.append(final_state)

    return winners, final_states