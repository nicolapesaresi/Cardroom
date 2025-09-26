import numpy as np
import copy
from cardroom.briscola.game.cards import Card
from cardroom.briscola.game.cards import FACES, SUITS


class BriscolaDealer:
    """Class for Briscola Dealer."""
    def __init__(self, seed: int|None = None):
        """Instantiates deck."""
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.initialize_deck()
        self.shuffle()
        self.set_briscola()

    def initialize_deck(self):
        """Generates a non-shuffled cards deck."""
        deck = []
        for suit_id in SUITS:
            for face_id in FACES:
                deck.append(Card(face_id, suit_id))

        self.deck = deck
        self.cards_left = len(self.deck)

    def shuffle(self):
        """Shuffles the deck in place."""
        np.random.shuffle(self.deck)

    def set_briscola(self) -> Card:
        """Sets the Briscola suit, as the suit of the last card in the deck, and updates rank of the briscolas in the deck."""
        self.spy = self.deck[0] # 0 beacause cards are dealt from last to first (pop)
        briscola_id = self.spy.suit_id

        for card in self.deck:
            if card.suit_id == briscola_id:
                card.rank += 100


    def get_cards_left(self) -> int:
        """Returns the number of the cards in the deck.
        Returns:
            number of cards left in the deck."""
        return len(self.deck)

    def deal(self) -> Card:
        """Removes and returns the last card in the deck.
        Returns:
            card"""
        if self.cards_left == 0:
            raise IndexError("Cannot deal a new card, deck is empty.")
        card = self.deck.pop()
        self.cards_left = self.get_cards_left()
        return card
    
    @staticmethod
    def draw_starting_player(n_players: int) -> int:
        """Randomly selects the starting player.
        Args:
            n_players: number of players in the game.
        Returns:
            index of the starting player [0,1]
        """
        return np.random.randint(n_players)
    
    def clone(self):
        """Creates a clone of the dealer, without any rendering attributes."""
        clone = BriscolaDealer(self.seed)
        clone.deck = [copy.copy(card) for card in self.deck]
        for card in clone.deck:
            if hasattr(card, "image"):
                del card.image
        clone.cards_left = self.cards_left
        clone.spy = copy.copy(self.spy)
        if hasattr(clone.spy, "image"):
            del clone.spy.image
        return clone



