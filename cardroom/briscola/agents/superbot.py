import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.bot import BotAgent

class SuperbotAgent(Agent):
    """Superbot agent class. Makes algotithmic decisions like Bot, but with perfect finals."""
    def __init__(self, name: str = "SuperbotAgent"):
        """Instantiates agent."""
        super().__init__(name)
        self.bot = BotAgent()

    @staticmethod
    def process_state(game_state: dict) -> dict:
        """Extract relevant informations from complete state, depending on the agent criteria.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            game_state: complete game state as returned from the env.
        """
        return game_state

    def select_action(self, game_state: dict, env: BriscolaEnv) -> int:
        """Makes a decision based on the processed state.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            action: index of the card in hand to be played.
        """
        hand = game_state["hand"]
        #hand = game_state["hands"]
        briscola_suit_id = game_state["briscola_id"]
        cards_on_table = game_state["cards_on_table"]
        cards_remaining = 40 - len(game_state["all_played_cards"])
        
        if cards_remaining > 6:
            action = self.bot.select_action(game_state)
        else:
            action, point_proj = self.optimal_play(env, game_state, cards_remaining)
            #print("Proiezione punti:  ", int(point_proj), " - ", 120 - int(point_proj))

        return action
    
    def make_card_sort_key(self, card_on_table, briscola_suit_id: int):
            def key(item):
                card = item[1]
                # scenario 1: carta in banco no carico, no briscola 
                if card_on_table.suit_id != briscola_suit_id and card_on_table.points < 10:
                    # Rule 1: prima ciò che strozza
                    if card.suit_id == card_on_table.suit_id and card.rank > card_on_table.rank and card.points > 0:
                        return (0, -card.rank)  # discendente -> -rank
                    # Rule 2: poi liscio non di briscola
                    elif card.suit_id != briscola_suit_id and card.points == 0:
                        return (1, -card.rank)
                    # Rule 3: poi briscole
                    elif card.suit_id == briscola_suit_id:
                        return (2, card.rank)  # ascendente
                    # Rule 4: tutto il resto
                    else:
                        return (3, card.rank)
                # scenario 2: carta in banco carico, no briscola
                elif card_on_table.suit_id != briscola_suit_id and card_on_table.points >= 10:
                    # Rule 1: prima ciò che strozza
                    if card.suit_id == card_on_table.suit_id and card.rank > card_on_table.rank:
                        return (0, -card.rank)  # discendente -> -rank
                    # Rule 2: poi briscole
                    elif card.suit_id == briscola_suit_id:
                        return (1, card.rank)  # ascendente
                    # Rule 3: tutto il resto
                    else:
                        return (2, card.rank)
                #scenario 3: carta in banco briscola
                else:
                    # Rule 1: liscio (precedenza no briscole)
                    if card.points == 0:
                        return (0, card.rank)
                    # Rule 2: se ho carichi gioco uso prima quelli di briscola
                    elif card.rank > card_on_table.rank:
                        return (1, card.rank)
                    else:
                        return (2, card.rank)
                
            return key
    
    def optimal_play(self, env: BriscolaEnv, state: dict, cards_remaining: int, root_player=None):
        """
        Computes the optimal move and outcome for the root player using recursion.

        Args:
            env: BriscolaEnv instance.
            state: dict, state from env.get_state().
            cards_remaining: int, number of cards left in the game.
            root_player: int | None, the player for whom we evaluate points 
                        (fixed at the first call).

        Returns:
            (best_action, final_points) tuple
            - best_action: index of the card in current player's hand to play
                        (None if terminal state)
            - final_points: total points of root_player at the end of the game
        """

        if root_player is None:
            root_player = state["current_player"]

        # Base case
        if state["done"] or cards_remaining == 0:
            return None, env.get_state()["points"][root_player]

        current_player = state["current_player"]
        #hand = state["hands"]
        hand = state["hand"]

        best_action = None
        maximizing = (current_player == root_player)

        best_value = -1 if maximizing else 121

        for i in range(len(hand)):
            # print("Livello: ", cards_remaining , "   iterazione: ", i, " valore: ", best_value, "enumerate: ", len(hand))
            # Clone environment
            env_clone = env.clone()
            env_clone.step(i)
            next_state = env_clone.get_observation()

            # Recursive evaluation
            _, value = self.optimal_play(env_clone, next_state, cards_remaining - 1, root_player)

            if maximizing:
                if value > best_value:
                    best_value = value
                    best_action = i
            else:
                if value < best_value:
                    best_value = value
                    best_action = i
        # print("Proiezione punti: ", best_value)
        return best_action, best_value
