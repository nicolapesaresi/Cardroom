import numpy as np
from cardroom.briscola.agents.agent import Agent

class BotAgent(Agent):
    """Bot agent class. Makes algotithmic decisions."""
    def __init__(self, name: str = "BotAgent"):
        """Instantiates agent."""
        super().__init__(name)

    @staticmethod
    def process_state(game_state: dict) -> list:
        """Extract relevant informations from complete state, depending on the agent criteria.
        RandomAgent only needs hand length to make a decision.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            processed_state: number of cards in hand.
        """
        return [game_state["hand"],game_state["briscola_id"], game_state["cards_on_table"]]

    def select_action(self, game_state: dict) -> int:
        """Makes a decision based on the processed state.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            action: index of the card in hand to be played.
        """
        hand = self.process_state(game_state)[0]
        briscola_suit_id = self.process_state(game_state)[1]
        cards_on_table = self.process_state(game_state)[2]

        if len(cards_on_table) == 0:
            # I play the lowest point,rank card
            action = min(enumerate(hand), key=lambda x: (x[1].points,x[1].rank))[0]
        else:
            indexed_hand = list(enumerate(hand))
            sorted_hand = sorted(indexed_hand, key=self.make_card_sort_key(cards_on_table[0], briscola_suit_id))
            action = sorted_hand[0][0]
        
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