import numpy as np
from cardroom.scopa.agents.agent import Agent

class RandomAgent(Agent):
    """Random agent class. Makes a decision randomly."""
    def __init__(self, name: str = "RandomAgent", seed: int|None = None):
        """Instantiates agent."""
        super().__init__(name)
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    @staticmethod
    def process_state(game_state: dict) -> int:
        """Extract relevant informations from complete state, depending on the agent criteria.
        RandomAgent only needs hand length to make a decision.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            processed_state: number of cards in hand.
        """
        n_cards = len(game_state["hand"])
        return n_cards

    def select_action(self, game_state: dict) -> int:
        """Makes a decision based on the processed state.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            action: index of the card in hand to be played.
        """
        n_cards = self.process_state(game_state)

        # random decision
        action = np.random.randint(n_cards)
        return action
