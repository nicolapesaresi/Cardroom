import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.algorithms.mcts import ISMCTS

class DonatelloAgent(Agent):
    """Donatello agent class. Makes a decision with random tree search.
    Args:
        name: agent name.
        simulations: number of MCTS simulation to run at every decision point.
        rollout_policy: policy for MCTS simulations.
        cpuct: exploration constant for MCTS.
        seed: optional seed for action selection.
    """
    def __init__(self, name: str = "Donatello", simulations = 100, rollout_policy:str = "random", cpuct:float = 1.4, seed: int|None = None):
        """Instantiates agent."""
        super().__init__(name)
        self.simulations = simulations
        self.rollout_policy = rollout_policy
        self.cpuct = cpuct
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

    def select_action(self, game_state: dict, env: BriscolaEnv, return_mcts: bool = False) -> int:
        """
        Makes a decision based on the processed state using Information-Set MCTS (ISMCTS).

        Args:
            game_state: complete game state as returned from the env.
            env: cloned environment for MCTS simulations.
            return_mcts: if True, also returns the root node of the ISMCTS tree and info_map

        Returns:
            action: index of the card in hand to be played.
            root_node (optional): root InfoSetNode of the ISMCTS tree.
            info_map (optional): info_map of the ISMCTS tree.
        """
        # Convert environment to ISMCTS root
        ismcts = ISMCTS(root_env=env, cpuct=self.cpuct, rollout_policy=self.rollout_policy)

        # Run simulations and select the best action
        action = ismcts.select_action(self.simulations)

        if return_mcts:
            root_node = ismcts.info_map[ismcts.root_key]
            return action, root_node, ismcts.info_map

        return action