import pygame
import os

CARD_IMAGES = os.path.join(os.path.dirname(__file__), "../../assets/cards")

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

POINTS = {
    1: 11,
    2: 0,
    3: 10,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 2,
    9: 3,
    10: 4,
}

RANK = {
    1: 10,
    2: 0,
    3: 9,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
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
        self.points = POINTS[face_id]
        self.rank = RANK[face_id]
        self.card_tuple = (self.face, self.suit)
        self.card_id = 10 * self.suit_id + self.face_id
        self.starting_x = 0
        self.starting_y = 120

        self.is_briscola = False

    def __str__(self):
        """Returns card string."""
        return f"{self.face} of {self.suit}"
    
    def __eq__(self, other):
        if isinstance(other, Card):
            return (self.suit_id == other.suit_id and
                    self.rank == other.rank and
                    self.points == other.points and
                    self.suit == other.suit)
        return False
    
    def get_image_path(self, style:str = "bergamasche"):
        """Returns the path of the image of the card.
        Args:
            style: style of the card (bergamasche, french, etc.)
        Returns:
            image_path: path of the image.
        """
        return f"{CARD_IMAGES}/{style}/{self.suit}_{self.face_id}.png"

    def load_image(self, width: int, height: int, style:str = "bergamasche"):
        """Loads the image for the card.
        Args:
            width: width of the card in pixels
            height: height of the card in pixels
            style: style of the card (bergamasche, french, etc.)
        """
        self.image = pygame.image.load(self.get_image_path(style))
        self.image = pygame.transform.scale(self.image, (width, height))

class CardRetro:
    """Class the loads the image for the back of a card."""
    def __init__(self):
        self.name = "retro"

    def load_image(self, width: int, height: int, style:str = "bergamasche"):
        """Loads the image for the card.
        Args:
            width: width of the card in pixels
            height: height of the card in pixels
            style: style of the card (bergamasche, french, etc.)
        """
        self.image = pygame.image.load(f"{CARD_IMAGES}/{style}/retro.png")
        self.image = pygame.transform.scale(self.image, (width, height))