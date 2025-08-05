import pygame
import yaml
import os
import sys
import numpy as np

from cardroom.briscola.cards.cards import Card, CardRetro

PYGAME_CONFIG = os.path.join(os.path.dirname(__file__), "render_config.yaml")

class BriscolaPygame:
    """Class for briscola pygame rendering."""
    def __init__(self):
        """Initializes the game window."""
        pygame.init()
        with open(PYGAME_CONFIG, "r") as f:
            config = yaml.safe_load(f)
        self.config = config["pygame"]
        # initialize random rotation for piles of taken cards
        self.random_x_offsets = [np.random.randint(-5,5) for _ in range(100)] # 100 allows different sequences for each player
        self.random_rotation = [np.random.randint(0,360) for _ in range(100)]

        self._initialize_screen()
        self.homepage()

    def _initialize_screen(self):
        """Initializes pygame screen."""
        config = self.config["display"]
        self.screen = pygame.display.set_mode((config["width"], config["height"]))
        pygame.display.set_caption(config["caption"])
        self.clickable_cards = [] # list of (rect: pygame.Rect, action: int) tuples
        self.running = True

    def homepage(self):
        """Loads the homepage and waits until start game is pressed. Then launchs starting animations."""
        self.screen.fill(self.config["colors"]["green"])
        font = pygame.font.SysFont(None, 48)
        
        start_text = font.render("Start Game", True, (255, 255, 255))
        start_rect = start_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 40))
        self.screen.blit(start_text, start_rect)
        
        settings_text = font.render("Settings", True, (255, 255, 255))
        settings_rect = settings_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 40))
        self.screen.blit(settings_text, settings_rect)

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    click_rect = pygame.Rect(pos[0], pos[1], 1, 1)

                    if start_rect.colliderect(click_rect):
                        return
                    elif settings_rect.colliderect(click_rect):
                        #self.settings()
                        pass
    
    def end_game(self, state):
        """Shows game results after game is finished.
        Args:
            state: final state of the game.    
        """
        self.screen.fill(self.config["colors"]["green"])
        font = pygame.font.SysFont(None, 25)
        
        p1 = state["player_names"][0]
        p2 = state["player_names"][1]
        points1 = state["points"][0]
        points2 = state["points"][1]
        if points1 == points2:
            result_str = "Draw"
        elif points1 > points2:
            result_str = f"{p1} wins."
        else:
            result_str = f"{p2} wins."

        p1_text = font.render(f"{p1}: {points1} points", True, (255, 255, 255))
        p1_rect = p1_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 40))
        self.screen.blit(p1_text, p1_rect)
        p2_text = font.render(f"{p2}: {points2} points", True, (255, 255, 255))
        p2_rect = p2_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 - 20))
        self.screen.blit(p2_text, p2_rect)

        result_text = font.render(result_str, True, (255, 255, 255))
        result_rect = result_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 20))
        self.screen.blit(result_text, result_rect)

        pygame.display.flip()

        # wait until a click to close
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                    self.close()
    
    def handle_events(self) -> int:
        """Handles events and actions.
        Returns:
            action: if player clicks with the mouse a clickable area.
        """
        # for event in pygame.event.get():
        #     # close game
        #     if event.type == pygame.QUIT:
        #         self.running = False
        #         self.close()
        while True:
            for event in pygame.event.get():
                # close game
                if event.type == pygame.QUIT:
                    self.running = False
                    self.close()
                # get action from mouse click
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    click_rect = pygame.Rect(pos[0], pos[1], 1, 1)
                    for rect, action in self.clickable_cards:
                        if rect.colliderect(click_rect):
                            return action

    def draw(self, state: dict):
        """Draws the screen at each call.
        Args:
            state: state of the game as returned from the env.
        """
        self.clickable_cards = [] # empty list at each rendering, it is set later by _draw_hand

        self.current_player_id = state["current_player"] # used in draw function to update clickable area with card rects
        self.screen.fill(self.config["colors"]["green"])
        self._draw_deck(state["deck_left"], state["briscola_spy"])
        for player_id in range(len(state["turn_order"])):
            self._draw_hand(player_id, state["hands"][player_id], state["player_names"][player_id], state["current_player"])
            self._draw_points(player_id, state["taken_cards"][player_id], state["points"][player_id])
        # draw card on the the table, depending on how many there are
        # this is because after the second player plays, the next state has 0 cards on the table
        if len(state["cards_on_table"]) == 1:
            cards_down = state["cards_on_table"]
            cards_order = state["turn_order"]
            self._draw_cards_on_table(cards_down, cards_order)
        elif len(state["cards_on_table"]) == 0:
            cards_down = state["all_played_cards"][-2:]
            cards_order = state["turn_history"][-2:]
            self._draw_cards_on_table(cards_down, cards_order)
        pygame.display.flip()
        
        if state["done"]:
            self.end_game(state)


    def close(self):
        """Closes the game window."""
        pygame.quit()
        sys.exit()            

    def _draw_hand(self, player_id: int, hand: list[Card], name: str, current_player_id: int):
        """Draws the hand and name of a player and updates clickable areas if player is current player.
        Args:
            player_id: id of the player, to know where to draw him.
            hand: hand of the player as returned by env.
            name: name of the player.
            current_player_id: id of current player. used to draw a circle next to his name.
        """
        if not hand:
            return
        
        screen_width = self.config["display"]["width"]
        screen_height = self.config["display"]["height"]
        config = self.config["cards"]
        spacing = config["hand_spacing"]
        card_width = config["width"]
        card_height = config["height"]
        style = config["style"]

        total_width = len(hand) * card_width + (len(hand) - 1) * spacing
        start_x = (screen_width - total_width) // 2

        if player_id == 0: # player 0 is South
            y = screen_height - card_height - config["side_margin"]
            name_y = y - config["name_margin"]
        elif player_id == 1: # player 1 is North
            y = config["side_margin"]
            name_y = y + card_height + config["name_margin"]

        # Render name and pointer
        font = pygame.font.SysFont(None, 24)
        name_surface = font.render(name, True, (255, 255, 255))
        name_rect = name_surface.get_rect(center=(screen_width // 2, name_y))
        self.screen.blit(name_surface, name_rect)
        if player_id == current_player_id:
            pointer_x = name_rect.left - config["pointer_margin"]
            pointer_y = name_rect.centery
            pygame.draw.circle(self.screen, (255, 255, 255), (pointer_x, pointer_y), config["pointer_radius"])

        # TODO: sort by suit
        for i, card in enumerate(hand):
            # load card image if it hasn't been loaded yet
            if not hasattr(card, "image"):
                card.load_image(card_width, card_height, style)

            x = start_x + i * (card_width + spacing)
            self.screen.blit(card.image, (x, y))
            # update clickable areas if current player
            rect = pygame.Rect(x, y, card_width, card_height)
            if player_id == current_player_id:
                self.clickable_cards.append((rect, i)) # rect is the area, i is the index of the card (the corresponding action)
    
    def _draw_cards_on_table(self, cards_on_table: list[Card], turn_order: list[int]):
        """Draws the cards on the table.
        Args:
            cards_on_table: list of the cards on the table.
            turn_order: order of play, to know who played each card.
        """
        if not cards_on_table:
            return
        
        screen_width = self.config["display"]["width"]
        screen_height = self.config["display"]["height"]
        config = self.config["cards"]
        card_width = config["width"]
        card_height = config["height"]
        style = config["style"]
        table_margin = config["table_margin"]

        x = (screen_width - card_width) // 2

        for i, card in enumerate(cards_on_table):
            if not hasattr(card, "image"):
                card.load_image(card_width, card_height, style)
            position = turn_order[i] # retrieve who played the card from turn order
            if position == 0: # played by South player
                y = screen_height // 2 + table_margin
            elif position == 1: # played by North player
                y = screen_height // 2 - table_margin - card_height
            self.screen.blit(card.image, (x, y))
        

    def _draw_deck(self, deck_left: int, briscola_spy: Card):
        """Draws the deck of cards on the table.
        Args:
            deck_left: number of cards remaining in the deck, including the briscola spy.
            briscola_spy: cards that decides the briscola, last in the deck.
        """
        if deck_left < 1:
            return

        screen_width = self.config["display"]["width"]
        screen_height = self.config["display"]["height"]
        config = self.config["cards"]
        card_width = config["width"]
        card_height = config["height"]
        style = config["style"]
        stack_offset = config["deck_offset"]  # pixels to offset each stacked card

        # Starting position (e.g., left of center table)
        x = config["side_margin"]
        y = screen_height // 2 - card_height // 2

        # draw briscola spy
        if not hasattr(briscola_spy, "image"):
            briscola_spy.load_image(card_width, card_height, style)
        rotated_image = pygame.transform.rotate(briscola_spy.image, 90)
        rotated_rect = rotated_image.get_rect(center=(2*x + card_width, y + card_height // 2))
        self.screen.blit(rotated_image, rotated_rect.topleft)
        # draw rest of the deck
        card_back = CardRetro()
        card_back.load_image(card_width, card_height, style)
        for i in range(deck_left-1):
            self.screen.blit(card_back.image, (x, y - i * stack_offset))

    def _draw_points(self, player_id: int, taken_cards: list[Card], points: int):
        """Draws the points and taken cards of a player.
        Args:
            player_id: id of the player, to know where to draw.
            taken_cards: list of player's taken cards as returned by env.
            points: points of the player.
        """
        if not taken_cards:
            return
        
        screen_width = self.config["display"]["width"]
        screen_height = self.config["display"]["height"]
        config = self.config["cards"]
        card_width = config["width"]
        card_height = config["height"]
        style = config["style"]
        margin = config["side_margin"]
        stack_offset = config["deck_offset"]  # pixels to offset each stacked card
        text_margin = config["name_margin"]

        x = (screen_width - card_height - margin*2 )

        if player_id == 0:  # South
            y = screen_height - card_height - margin * 2
            text_y = screen_height - text_margin * 2 - card_height - margin * 2
        elif player_id == 1:  # North
            y = margin * 2
            text_y = text_margin * 2 + card_height + margin * 2

        # write points text
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(f"Points: {points}", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(x, text_y))
        self.screen.blit(text_surface, text_rect)
        # draw taken cards pile
        card_back = CardRetro()
        card_back.load_image(card_width, card_height, style)
        for i in range(len(taken_cards)):
            rotated_image = pygame.transform.rotate(card_back.image, self.random_rotation[i + player_id])
            random_x = x - card_height//2 + self.random_x_offsets[i + player_id]
            offset_y = y - (i // 10) * stack_offset
            self.screen.blit(rotated_image, (random_x, offset_y))

    # def settings(self):
    #     """Loads the Settings page and waits until back button is pressed."""
    #     settings_options = self.config["settings_options"]

    #     # Current indices for each option
    #     cards_index = settings_options["card_styles"].index(self.config["cards"]["style"])
    #     p1_index = settings_options["default_p1_mode"]
    #     p2_index = settings_options["default_p2_mode"]
    #     show_cards_index = settings_options["default_show"]

    #     # Fonts
    #     font = pygame.font.SysFont(None, 48)
    #     back_font = pygame.font.SysFont(None, 30)
    #     button_font = pygame.font.SysFont(None, 36)

    #     w, h = self.screen.get_width(), self.screen.get_height()

    #     def draw_settings():
    #         self.screen.fill(self.config["colors"]["green"])

    #         # Labels
    #         cards_text = font.render("Cards:", True, (255, 255, 255))
    #         cards_rect = cards_text.get_rect(center=(w // 4, h // 4))
    #         self.screen.blit(cards_text, cards_rect)

    #         players_text = font.render("Players:", True, (255, 255, 255))
    #         players_rect = players_text.get_rect(center=(w // 4, h // 4 * 2))
    #         self.screen.blit(players_text, players_rect)

    #         show_cards_text = font.render("Show hole cards:", True, (255, 255, 255))
    #         show_cards_rect = show_cards_text.get_rect(center=(w // 4, h // 4 * 3))
    #         self.screen.blit(show_cards_text, show_cards_rect)

    #         # Options
    #         cards_option = settings_options["card_styles"][cards_index]
    #         p1_option = settings_options["player_mode"][p1_index]
    #         p2_option = settings_options["player_mode"][p2_index]
    #         show_cards_option = settings_options["show_hole_cards"][show_cards_index]

    #         cards_opt_text = back_font.render(cards_option, True, (255, 255, 255))
    #         p1_opt_text = back_font.render(p1_option, True, (255, 255, 255))
    #         p2_opt_text = back_font.render(p2_option, True, (255, 255, 255))
    #         show_cards_opt_text = back_font.render(show_cards_option, True, (255, 255, 255))

    #         cards_opt_rect = cards_opt_text.get_rect(center=(w // 2 + 10, h // 4))
    #         p1_opt_rect = p1_opt_text.get_rect(center=(w // 2 + 10, h // 4 * 2 + 10))
    #         p2_opt_rect = p2_opt_text.get_rect(center=(w // 2 + 10, h // 4 * 2 - 10))
    #         show_cards_opt_rect = show_cards_opt_text.get_rect(center=(w // 2 + 10, h // 4 * 3))

    #         self.screen.blit(cards_opt_text, cards_opt_rect)
    #         self.screen.blit(p1_opt_text, p1_opt_rect)
    #         self.screen.blit(p2_opt_text, p2_opt_rect)
    #         self.screen.blit(show_cards_opt_text, show_cards_opt_rect)

    #         # Back button
    #         back_text = back_font.render("back", True, (255, 255, 255))
    #         back_rect = back_text.get_rect(center=(w // 2, h // 8 * 7))
    #         self.screen.blit(back_text, back_rect)

    #         # Buttons
    #         button_color = (200, 200, 200)
    #         self.cards_button_rect = pygame.Rect(w // 4 * 3, h // 4 - 20, 40, 40)
    #         self.p1_button_rect = pygame.Rect(w // 4 * 3, h // 4 * 2 - 20 + 10, 40, 40)
    #         self.p2_button_rect = pygame.Rect(w // 4 * 3, h // 4 * 2 - 20 - 10, 40, 40)
    #         self.show_cards_button_rect = pygame.Rect(w // 4 * 3, h // 4 * 3 - 20, 40, 40)

    #         for btn_rect in [self.cards_button_rect, self.p1_button_rect, self.p1_button_rect, self.show_cards_button_rect]:
    #             pygame.draw.rect(self.screen, button_color, btn_rect)
    #             label = button_font.render("[>]", True, (0, 0, 0))
    #             label_rect = label.get_rect(center=btn_rect.center)
    #             self.screen.blit(label, label_rect)

    #         pygame.display.flip()
    #         return back_rect  # needed to check back click

    #     back_rect = draw_settings()

    #     # Main loop
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 self.close()
    #             elif event.type == pygame.MOUSEBUTTONDOWN:
    #                 pos = pygame.mouse.get_pos()
    #                 click_rect = pygame.Rect(pos[0], pos[1], 1, 1)

    #                 if back_rect.colliderect(click_rect):
    #                     return

    #                 elif self.cards_button_rect.colliderect(click_rect):
    #                     cards_index = (cards_index + 1) % len(settings_options["card_styles"])
    #                     back_rect = draw_settings()

    #                 elif self.p1_button_rect.colliderect(click_rect):
    #                     p1_index = (p1_index + 1) % len(settings_options["player_mode"])
    #                     back_rect = draw_settings()
    #                 elif self.p2_button_rect.colliderect(click_rect):
    #                     p2_index = (p2_index + 1) % len(settings_options["player_mode"])
    #                     back_rect = draw_settings()

    #                 elif self.show_cards_button_rect.colliderect(click_rect):
    #                     show_cards_index = (show_cards_index + 1) % len(settings_options["show_hole_cards"])
    #                     back_rect = draw_settings()