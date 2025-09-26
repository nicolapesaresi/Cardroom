from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.agents.bot import BotAgent


if __name__ == "__main__":
    human = HumanAgent()
    random = RandomAgent()
    donatello = DonatelloAgent()
    bot = BotAgent()
    
    names = ["Nick", "Donatello"]
    agents = [human, random]
    env = BriscolaEnv(names, "text")

    while not env.done:
        player_id = env.current_player_id
        player = env.players[player_id]
        agent = agents[player_id]
        action = agent.select_action(env.get_observation())
        env.step(action)