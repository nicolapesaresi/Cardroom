import argparse

from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_games", help="number of games to play")
    args = parser.parse_args()
    n = int(args.n_games)
    
    human = BriscolaPlayer(HumanAgent(), name="Nicola")
    random = BriscolaPlayer(RandomAgent(), name="Mike")
    random2 = BriscolaPlayer(RandomAgent(), name="John")
    #players = [human, random]
    players = [random, random2]

    results = []
    for i in range(n):
        env = BriscolaEnv(players, "text")
        while not env.done:
            player = env.players[env.current_player_id]
            action = player.get_action(env.get_observation())
            env.step(action)
        
        result = env.result
        results.append(result)
    
    # summary
    print("Summary:")
    print(f"{players[0].name} wins: {results.count(1)}")
    print(f"{players[1].name} wins: {results.count(-1)}")
    print(f"Draws: {results.count(0)}")