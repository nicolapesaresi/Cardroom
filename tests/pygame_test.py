from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent

if __name__ == "__main__":
    human = BriscolaPlayer(HumanAgent(input_mode="pygame"), name="Nicola")
    human2 = BriscolaPlayer(HumanAgent(input_mode="pygame"), name="EvilNick")
    random = BriscolaPlayer(RandomAgent(), name="Mike")
    random2 = BriscolaPlayer(RandomAgent(), name="John")
    players = [human, human2]
    #players = [random, random2]

    env = BriscolaEnv(players, "pygame")

    while not env.done:
        player = env.players[env.current_player_id]
        action = player.get_action(env.get_observation())
        env.step(action)