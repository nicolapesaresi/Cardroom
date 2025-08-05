from cardroom.briscola.render.pygame import BriscolaPygame
from cardroom.briscola.agents.agent import Agent

INPUT_MODES = ["text", "pygame"]

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
            return self._text_select_action(hand)
        elif self.input_mode == "pygame":
            return self._pygame_select_action(hand)

    def _text_select_action(self, hand: list) -> int:
        """Queries human for text input.
        Args:
            hand: cards in hand.
        Returns:
            action: index of the card in hand to be played.
        """
        n_cards = len(hand)
        valid_actions = list(range(n_cards))
        print(f"Cards in hand: {', '.join(str(card) for card in hand)}")

        while True:
            user_input = input(f"Select a card: [{'-'.join(str(i) for i in valid_actions)}] ")
            try:
                action = int(user_input)
                if action in valid_actions:
                    break
                else:
                    print(f"Invalid choice. Valid actions: [{'-'.join(str(i) for i in valid_actions)}]")
            except ValueError:
                print("Please enter a valid number.")
        return action
    
    def _pygame_select_action(self, hand: list) -> int:
        """Retrieves action from pygame mouse click.
        Returns:
            action: index of the card in hand to be played.
        """
        action = self.pygame.handle_events()
        return action

    def set_pygame_action_retriever(self, pygame_renderer: BriscolaPygame):
        """Sets a pygame renderer object to pass the actions via the handle_events method."""
        self.pygame = pygame_renderer