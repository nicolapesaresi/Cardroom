import argparse
from tqdm import tqdm
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.donatello import DonatelloAgent

from cardroom.briscola.utils.scoring import play_n_games

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_games", help="number of games to play")
    parser.add_argument("oppo", help="opponent, random or bot")
    parser.add_argument("render_mode", help="text, pygame or None")
    args = parser.parse_args()
    n_games = int(args.n_games)
    oppo = args.oppo
    render_mode = args.render_mode
    if render_mode == "None":
        render_mode = None
    
    #human = HumanAgent(input_mode=render_mode)
    random = RandomAgent()
    donatello = DonatelloAgent(simulations=200)
    bot = BotAgent()

    if oppo == "random":
        oppagent = random
    elif oppo == "bot":
        oppagent = bot
    else:
        raise NotImplementedError("oppo must be one of ['random', 'bot']")

    results = []

    for i in tqdm(range(n_games)):
        agents = [donatello, oppagent]
        env = BriscolaEnv([agent.name for agent in agents], render_mode=render_mode)
 
        winners, states = play_n_games(agents, n_games, render_mode)