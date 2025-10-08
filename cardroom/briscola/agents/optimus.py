import numpy as np
import math
import itertools
import copy
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.game.cards import Card, FACES, SUITS

class OptimusAgent(Agent):
    """Optimus agent class. Plays like bot for the first [40 - depth] cards
    remaining. Makes best decisions in expected value starting from [depth] cards remaining."""
    def __init__(self, name: str = "OptimusAgent", depth: int = 8):
        """Instantiates agent."""
        super().__init__(name)
        self.depth = depth

    @staticmethod
    def process_state(game_state: dict) -> dict:
        """Extract relevant informations from complete state, depending on the agent criteria.
        Args:
            game_state: complete game state as returned from the env.
        Returns:
            processed_state: game state.
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
        
        if cards_remaining <= 6:
            action, point_proj = self.optimal_play(env, game_state, cards_remaining)
        #plays like a bot
        elif cards_remaining > self.depth:
            if len(cards_on_table) == 0:
                # I play the lowest point,rank card
                action = min(enumerate(hand), key=lambda x: (x[1].points,x[1].rank))[0]
            else:
                indexed_hand = list(enumerate(hand))
                sorted_hand = sorted(indexed_hand, key=self.make_card_sort_key(cards_on_table[0], briscola_suit_id))
                action = sorted_hand[0][0]
        else:
            action, point_proj = self.best_expected_value_play(env, game_state, cards_remaining)

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
            #print(root_player, env.players[root_player].name, env.get_state()["points"][root_player])
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

    def best_expected_value_play(self, env: BriscolaEnv, state: dict, cards_remaining: int, root_player=None):
        """
        Computes the optimal move (in expected value) and outcome for the root player using recursion.

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
            - final_points: total expected points of root_player at the end of the game
        """

        if root_player is None:
            root_player = state["current_player"]
        
        #base case
        if cards_remaining <= 6:
            best_action, best_value = self.optimal_play(env, state, cards_remaining, root_player)
        else:
            current_player = state["current_player"]
            hand = state["hand"]      
            # I concatenate all the unknown cards, that could be either in oppo's hand or in deck or on table
            full_deck = []
            for suit_id in SUITS:
                for face_id in FACES:
                    full_deck.append(Card(face_id, suit_id))
            
            for card in full_deck:
                if card.suit_id == state["briscola_spy"].suit_id:
                    card.rank += 100
            
            unknown_cards = [copy.copy(card) for card in full_deck]
            for x in full_deck:
                for y in [state["taken_cards"][0],state["taken_cards"][1],[state["briscola_spy"]],state["hand"],state["cards_on_table"]]:
                    for z in y:
                        if x.suit_id == z.suit_id and x.face_id == z.face_id:
                            unknown_cards.remove(x)

            # n : number of unknown cards that are in deck
            # m : number of unknown cards that are i oppo's hand
            N = len(unknown_cards)
            n = max(40 - 6 - 2*(state["turn_counter"] - 1) - 1, 0)
            m = N - n

            total_count = math.factorial(N) / math.factorial(m)

            best_action = None
            maximizing = (current_player == root_player)
            best_value = -1 if maximizing else 121

            for i in range(len(hand)):
                total_value = 0
                for cards in itertools.permutations(unknown_cards, n):
                    new_deck = list(cards)
                    #aggiungo la briscola all'inizio
                    new_deck.insert(0,state["briscola_spy"])
                    new_oppo_hand = [x for x in unknown_cards if x not in cards]
                    # Clone environment
                    env_clone = env.clone()

                    # I replace deck and hand with the new permutations
                    env_clone.players[1 - current_player].hand = [x for x in new_oppo_hand]
                    env_clone.dealer.deck = [x for x in new_deck]
                    env_clone.step(i)
                    next_state = env_clone.get_observation()

                    # Recursive evaluation
                    _, partial_value = self.best_expected_value_play(env_clone, next_state, cards_remaining - 1, root_player)
                    total_value = total_value + partial_value

                value = total_value / total_count

                if maximizing:
                    if value > best_value:
                        best_value = value
                        best_action = i
                else:
                    if value < best_value:
                        best_value = value
                        best_action = i

        return best_action, best_value

