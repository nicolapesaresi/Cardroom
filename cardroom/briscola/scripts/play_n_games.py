import argparse
from tqdm import tqdm
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.donatello import DonatelloAgent

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
    donatello = DonatelloAgent(simulations=100)
    bot = BotAgent()

    if oppo == "random":
        oppagent = random
        oppname = "Random"
    elif oppo == "bot":
        oppagent = bot
        oppname = "Bot"
    else:
        raise NotImplementedError("oppo must be one of ['random', 'bot']")

    names = ["Donatello", oppname]
    results = []

    for i in tqdm(range(n_games)):
        env = BriscolaEnv(names, render_mode=render_mode)
        # if render_mode == "pygame":
        #     human.set_pygame_action_retriever(env.pygame)
        agents = [donatello, oppagent]
        
        while not env.done:
            player_id = env.current_player_id
            player = env.players[player_id]
            agent = agents[player_id]
            if isinstance(agent, DonatelloAgent): # ugly. should be the same call for all agents
                action = agent.select_action(env.get_observation(), env.clone_from_observation())
            else:
                action = agent.select_action(env.get_observation())
            env.step(action)
        
        result = env.result
        results.append(result)

        # partial summary
        if i % (n_games // 10) == 0:
            print("Partial Summary:")
            print(f"{names[0]} wins: {results.count(1)}")
            print(f"{names[1]} wins: {results.count(-1)}")
            print(f"Draws: {results.count(0)}")
    
    # summary
    print("Final Summary:")
    print(f"{names[0]} wins: {results.count(1)}")
    print(f"{names[1]} wins: {results.count(-1)}")
    print(f"Draws: {results.count(0)}")