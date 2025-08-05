FACES = {
    1: "A",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "J",
    9: "Q",
    10: "K",
}

SUITS = {
    0: "Ori",
    1: "Coppe",
    2: "Spade",
    3: "Bastoni",
}

class Card:
    """Class for a card of the italian deck."""
    def __init__(self, face_id: int, suit_id: int):
        """Instantiates a card.
        Args:
            face_id: id of face number of the card.
            suit_id: id of suit of the card.
        """
        if face_id not in FACES.keys():
            raise KeyError(f"{face_id} not a valid ID of card face.")
        if suit_id not in SUITS.keys():
            raise KeyError(f"{suit_id} not a valid ID of card suit.")
        
        self.face_id = face_id
        self.suit_id = suit_id

        self.face = FACES[face_id]
        self.suit = SUITS[suit_id]
        self.card_tuple = (self.face, self.suit)

        self.is_briscola = False

    def __str__(self):
        """Returns card string."""
        return f"{self.face} of {self.suit}"