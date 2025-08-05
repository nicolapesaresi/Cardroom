from cardroom.scopa.agents.agent import Agent

INPUT_MODES = ["text"]

class HumanAgent(Agent):
    """Human agent class. Asks input for a decision."""
    def __init__(self, name: str = "HumanAgent", input_mode = "text"):
        """Instantiates agent."""
        super().__init__(name)
        if input_mode not in INPUT_MODES:
            raise NotImplementedError({input_mode})
        self.input_mode = input_mode
        
    @staticmethod
    def process_state(game_state: dict) -> int:
        """Extract relevant informations from complete state, depending on the agent criteria.
        HumanAgent needs the cards in hand.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            processed_state: placeholder.
        """
        processed_state = game_state["hand"]
        return processed_state

    def select_action(self, game_state: dict) -> int:
        """Makes a decision based on the processed state.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            action: index of the card in hand to be played.
        """
        hand = self.process_state(game_state)

        # human decision
        if self.input_mode == "text":
            n_cards = len(hand)
            valid_actions = list(range(n_cards))
            print(f"Cards in hand: {', '.join(str(card) for card in hand)}")
            action = int(input(f"Select a card: [{'-'.join(str(action_idx) for action_idx in valid_actions)}] "))
            while action not in valid_actions:
                action = int(input(f"Invalid choice. Valid actions: [{'-'.join(str(action_idx) for action_idx in valid_actions)}] "))    
        return action
