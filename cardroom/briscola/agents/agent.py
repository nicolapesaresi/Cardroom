from abc import ABC, abstractmethod

class Agent(ABC):
    """Abstract base class for agents."""
    def __init__(self, name: str = "Agent"):
        self.name = name

    @staticmethod
    @abstractmethod
    def process_state(game_state: dict):
        """Process the game state to extract the infomration needed for the agent to make a decision.
        Args:
            game_state: complete game state as returned from the env.
            
        Returns:
            processed_state: input for the agent to make a decision."""
        pass

    @abstractmethod
    def select_action(self, game_state: dict) -> int:
        """Process the game state and select an action.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            action: index of the card in hand to be played.
        """
        pass