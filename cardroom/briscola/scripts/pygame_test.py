from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.agents.bot import BotAgent

if __name__ == "__main__":
    human = HumanAgent(input_mode="pygame")
    random = RandomAgent()
    donatello = DonatelloAgent(simulations=5000)
    bot = BotAgent()

    names = ["Nick", "Donatello"]
    agents = [human, donatello]
    env = BriscolaEnv(names, "pygame")

    human.set_pygame_action_retriever(env.pygame) #TODO: this should probably be handled by an orchestrator

    while not env.done:
        player_id = env.current_player_id
        agent = agents[player_id]
        if isinstance(agent, DonatelloAgent): # ugly. should be the same call for all agents
            action = agent.select_action(env.get_observation(), env.clone_from_observation())
        else:
            action = agent.select_action(env.get_observation())
        env.step(action)