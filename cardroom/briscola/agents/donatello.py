import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.algorithms.mcts import ISMCTS

class DonatelloAgent(Agent):
    """Donatello agent class. Makes a decision with random tree search."""
    def __init__(self, name: str = "Donatello", simulations = 100, seed: int|None = None):
        """Instantiates agent."""
        super().__init__(name)
        self.simulations = simulations
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    @staticmethod
    def process_state(game_state: dict) -> int:
        """Extract relevant informations from complete state, depending on the agent criteria.
        Args:
            game_state: complete game observation as returned from the env.
        Returns:
            processed_state: number of cards in hand.
        """
        proc_state = game_state #UPDATE
        return proc_state
    
    # @staticmethod
    # def get_legal_actions(game_state: dict) -> list:
    #     """Retrieves legal actions froms state."""

    def select_action(self, game_state: dict, env: BriscolaEnv, return_root: bool = False) -> int:
        """
        Makes a decision based on the processed state using Information-Set MCTS (ISMCTS).

        Args:
            game_state: complete game state as returned from the env.
            env: cloned environment for MCTS simulations.
            return_root: if True, also returns the root node of the ISMCTS tree.

        Returns:
            action: index of the card in hand to be played.
            root_node (optional): root InfoSetNode of the ISMCTS tree.
        """
        # Convert environment to ISMCTS root
        #TODO: pass here the parameters, cpuct and simulations, maybe from self
        # ismcts = ISMCTS(root_env=env, cpuct=self.cpuct, rollout_policy=self.rollout_policy)
        ismcts = ISMCTS(root_env=env, rollout_policy="bot")

        # Run simulations and select the best action
        action = ismcts.select_action(self.simulations)

        if return_root:
            root_node = ismcts.info_map[ismcts.root_key]
            return action, root_node

        return action