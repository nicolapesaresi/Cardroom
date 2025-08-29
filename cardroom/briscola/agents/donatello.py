import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.algorithms.mcts import MCTSNode

class DonatelloAgent(Agent):
    """Donatello agent class. Makes a decision with random tree search."""
    def __init__(self, name: str = "Donatello", seed: int|None = None):
        """Instantiates agent."""
        super().__init__(name)
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

    def select_action(self, game_state: dict, env: BriscolaEnv) -> int:
        """Makes a decision based on the processed state.
        Args:
            game_state: complete game state as returned from the env.
            env: cloned environment for MCTS simulations.
        Returns:
            action: index of the card in hand to be played.
        """
        n_cards = self.process_state(game_state)

        root_node = MCTSNode(game_state, env)
        action = root_node.best_action(simulations=3)

        return action

