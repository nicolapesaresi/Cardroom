from cardroom.scopa.cards.cards import Card
from cardroom.scopa.agents.agent import Agent

class ScopaPlayer:
    """Class for Scopa Player."""
    def __init__(self, agent: Agent, name: str = "Player"):
        """Instantiates scopa player.
        Args:
            agent: agent that makes the decisions for the player.
            name: name of the player.
        """
        self.agent = agent
        self.name = name
        # uncomment if score tracking is implemented
        # self.games_played = 0
        # self.wins = 0
        self.reset()

    def reset(self):
        """Resets player to initial state, ready to start a new game."""
        self.taken = [] # list of taken cards, used to count points
        self.points = 0
        self.hand = []

    def get_action(self, game_state: dict) -> int:
        """Calls the agent to decide a move, then returns the selected action.
        Args:
            game_state: dictionary describing the state of the game.
        Returns:
            chosen action
        """
        if len(self.hand) == 0:
            raise IndexError(f"{self.name}'s hand is empty when play_card is called.")
        return self.agent.select_action(game_state)

    def play_card(self, action: int) -> Card:
        """Removes and returns a card from the player hand.
        Args:
            action: action to perform, selected by the agent.
        Returns:
            chosen card
        """
        if not isinstance(action, int):
            raise TypeError(f"card index must be an integer, instead got {type(action)}.")
        if action > len(self.hand):
            raise IndexError(f"{self.name}'s hand only has {len(self.hand)} cards, instead got index {action}.")
        return self.hand.pop(action)