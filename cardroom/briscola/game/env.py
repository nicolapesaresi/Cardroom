import pygame

from copy import deepcopy
from cardroom.briscola.game.dealer import BriscolaDealer
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.render.pygame import BriscolaPygame

RENDER_MODES = [None, "text", "pygame"]

class BriscolaEnv:
    """Environment for a game of Briscola."""
    def __init__(self, names: list[str,str]|None, render_mode: str = "text"):
        """Instantiates the environment.
        Args:
            names: names of players who are going to play the game.
            render_mode: render mode for the game.
        """
        self.n_players = 2
        if render_mode not in RENDER_MODES:
            raise NotImplementedError(f"Render mode {render_mode} not implented.")
        self.render_mode = render_mode
        self.instantiate_players(names)
        self.reset()
        self.render()

    def instantiate_players(self, names: list[str, str] | None = None):
        """Instantiates the players for the game.
        Args:
            players: list of players who are going to play the game.
        """
        if names is None:
            self.players = [BriscolaPlayer(f"P{i}") for i in range(self.n_players)]
        else:
            if len(names) != self.n_players:
                raise ValueError(f"Expected {self.n_players} players, got {len(names)}")
            self.players = [BriscolaPlayer(name) for name in names]

    def set_turn_order(self, first_id: int):
        """Sets the turn order for the players given who has to start.
        Args:
            first_id: index of the player who has to start.
        """
        if first_id < 0 or first_id >= self.n_players:
            raise ValueError(f"Invalid player index {first_id}, must be in [0, {self.n_players - 1}]")
        default_order = list(range(self.n_players))
        self.turn_order = default_order[first_id:] + default_order[:first_id]


    def reset(self):
        """Resets the env to the initial state at the beginning of a new game."""
        self.done = False
        self.played_cards_history = []
        self.turn_history = []
        
        # clean up players and dealer
        self.dealer = BriscolaDealer()
        for player in self.players:
            player.reset()
        
        # set up new game
        self.briscola_spy = self.dealer.spy
        self.briscola_suit_id = self.briscola_spy.suit_id
        self.current_player_id = self.dealer.draw_starting_player(self.n_players)
        self.set_turn_order(self.current_player_id)
        # deal first 3 cards
        for id in self.turn_order:
            player = self.players[id]
            for _ in range(3):
                player.hand.append(self.dealer.deal())
        self.turn_counter = 1
        self.cards_on_table = []

    def pass_turn(self):
        """Passes the turn to the next player."""
        self.current_player_id = (self.current_player_id + 1) % self.n_players

    def resolve_turn(self):
        """Updates the game after a whole turn has been played."""
        if len(self.cards_on_table) != self.n_players:
            raise ValueError(f"Cannot resolve turn, expected {self.n_players} cards on table but got {len(self.cards_on_table)}.")
        # first card suit commands, without a briscola
        first_suit_id = self.cards_on_table[0].suit_id
        # judge winner and update points
        played_rank = []
        played_points = 0
        for card in self.cards_on_table:
            effective_rank = card.rank
            if card.suit_id == first_suit_id:
                effective_rank += 50
            played_rank.append(effective_rank)
            played_points += card.points
        winning_card_idx = played_rank.index(max(played_rank))
        winning_player_id = self.turn_order[winning_card_idx]
        self.players[winning_player_id].points += played_points
        self.players[winning_player_id].taken_cards.extend(self.cards_on_table)
        # update turn order
        self.current_player_id = winning_player_id
        self.set_turn_order(winning_player_id)
        # reset cards on table
        self.cards_on_table = []

    def get_state(self) -> dict:
        """Retrieves the current game state.
        Returns:
            state: current game state.
        """
        state = {}
        state["player_names"] = [player.name for player in self.players]
        state["current_player"] = self.current_player_id
        state["turn_order"] = self.turn_order
        state["turn_counter"] = self.turn_counter
        state["deck_left"] = self.dealer.cards_left
        state["cards_on_table"] = self.cards_on_table
        state["briscola_id"] = self.briscola_suit_id
        state["briscola_spy"] = self.briscola_spy
        state["hands"] = [player.hand for player in self.players]
        state["points"] = [player.points for player in self.players]
        state["taken_cards"] = [player.taken_cards for player in self.players]
        state["all_played_cards"] = self.played_cards_history
        state["turn_history"] = self.turn_history
        state["done"] = self.done

        return state
    
    def get_observation(self) -> dict:
        """Retrieves observation of game state for current player.
        Returns:
            obs: observation of the game state for the current player.
        """
        state = self.get_state().copy()
        obs = {}

        # hide other player hand from player obs
        for key in state:
            if key in ["hands", "points"]:
                continue
            else:
                obs[key] = state[key]

        obs["hand"] = state["hands"][self.current_player_id]
        obs["current_player_points"] = state["points"].pop(self.current_player_id)
        obs["other_player_points"] = state["points"].pop()
        return obs

    def step(self, action: int):
        """Executes the action of the current player and updates the state of the game.
        Args:
            action: index of the card to be played by the current player."""
        # check if current player if last to play
        if self.current_player_id == self.turn_order[-1]:
            is_last = True
        else:
            is_last = False
        # get current player
        player = self.players[self.current_player_id]
        # play his action
        card = player.play_card(action)
        self.cards_on_table.append(card)
        self.played_cards_history.append(card)
        self.turn_history.append(self.current_player_id)

        # if he's last, resolve the turn, else just pass the turn
        if not is_last:
            self.pass_turn()
        else:
            self.resolve_turn()
            self.turn_counter += 1
            # check if game is over, else deal new cards
            if self.turn_counter > 20:
                self.result, points = self.check_result()
                self.done = True
            else:
                if self.dealer.cards_left > 0:
                    for id in self.turn_order:
                        player = self.players[id]
                        player.hand.append(self.dealer.deal())
        self.render()

    def check_result(self) -> tuple[int, list[int, int]]:
        """Checks result of the game.
        Returns:
            result : 0 if a draw, 1 if p0 won, -1 if p1 won.
            points: list of points of the players.
        """
        points = []
        for player in self.players:
            points.append(player.points)
        if points[0] == points[1]:
            result = 0
        elif points[0] > points[1]:
            result = 1
        else:
            result = -1
        result = result
        return result, points
    
    def get_legal_actions(self) -> list:
        """Returns legal actions at the current state of the game.
        Returns:
            legal_actions: list of legal actions for the current player.
        """
        obs = self.get_observation()
        return list(range(len(obs["hand"])))

    def render(self):
        """Renders the current state of the game."""
        state = self.get_state()

        if self.render_mode is None:
            return # don't render
        elif self.render_mode == "text":
            self.text_render(state)
        elif self.render_mode == "pygame":
            self.pygame_render(state)

    def text_render(self, state):
        """Renders in text mode.
        Args:
            state: current state of the game.
        """
        # print beginning of the game
        if state["turn_counter"] == 1 and len(state["cards_on_table"]) == 0:
            print("New game starting!")
            print(f"Players: {[player.name for player in self.players]}")
            print(f"{self.players[self.current_player_id].name} to start.")
            print(f"Briscola: {self.briscola_spy.suit} - Spy: {self.briscola_spy}")
            self.last_rendered_turn = 1
            return
        # print turn start
        if len(state["cards_on_table"]) == 1:
            print(state["points"])
            print(f"---TURN {state['turn_counter']}---")
        # print last action
        print(f"{self.players[state['turn_history'][-1]].name} plays {state['all_played_cards'][-1]}.")
        # print end of the game
        if state["done"]:
            print("Game finished.")
            for player in self.players:
                print(f"{player.name}: {player.points} points.")
            if state["points"][0] == state["points"][1]:
                print("It's a draw!")
            elif state["points"][0] > state["points"][1]:
                print(f"{self.players[0].name} wins!")
            else:
                print(f"{self.players[1].name} wins!")

    def pygame_render(self, state):
        """Renders in pygame mode.
        Args:
            state: current state of the game.
        """
        # also render text for logging
        self.text_render(state)
        
        # init pygame
        if not pygame.get_init():
            self.pygame = BriscolaPygame()
            # for player in self.players:
            #     #if isinstance(player.agent, HumanAgent) and player.agent.input_mode ==  "pygame":
            #     # TODO: at the moment only humans players are handled
            #     player.agent.set_pygame_action_retriever(self.pygame)

        if self.pygame.running:
            # self.pygame.handle_events() is called by human agent select_action
            self.pygame.draw(state)
        else:
            self.pygame.close()


    def clone(self):
        """Creates a new instance of the enviroment, which is a copy of the current state.
        Returns:
            clone: clone of the current environment.
        """
        # make a fresh env with the same players and no render
        clone = BriscolaEnv([p.name for p in self.players], render_mode=None)

        clone.n_players = self.n_players
        clone.done = self.done
        for i in range(len(clone.players)):
            real_player = self.players[i]
            clone.players[i].hand = [card for card in real_player.hand]
            for card in clone.players[i].hand:
                if hasattr(card, "image"):
                    del card.image
        clone.played_cards_history = list(self.played_cards_history)
        for card in clone.played_cards_history:
            if hasattr(card, "image"):
                del card.image
        clone.turn_history = list(self.turn_history)
        clone.current_player_id = self.current_player_id
        clone.turn_order = list(self.turn_order)
        clone.turn_counter = self.turn_counter
        clone.cards_on_table = list(self.cards_on_table)
        for card in clone.cards_on_table:
            if hasattr(card, "image"):
                del card.image

        # clone dealer
        clone.dealer = self.dealer.clone()

        # clone players (but not their agents/rendering)
        clone.players = []
        for orig in self.players:
            new_p = BriscolaPlayer(orig.name)
            new_p.hand = list(orig.hand)
            new_p.points = orig.points
            new_p.taken_cards = list(orig.taken_cards)
            clone.players.append(new_p)

        # briscola info
        clone.briscola_spy = self.briscola_spy
        clone.briscola_suit_id = self.briscola_suit_id

        return clone

    
    def clone_from_observation(self):
        """Creates a copy of the environment from the side of one player, so randomly shuffling remaining unknown cards.
        Returns:
            obs_clone: random clone of the environment from point of view of one player.
        """
        obs = self.get_observation()
        clone = self.clone()

        # mix deck and other player hand
        if self.dealer.cards_left > 0: # if deck is empty, don't shuffle because other player hand is known
            hand_sizes = []
            briscola_spy = clone.dealer.deck.pop(0)
            for player_id in range(len(clone.players)):
                hand_sizes.append(len(clone.players[player_id].hand))
                if player_id == clone.current_player_id:
                    continue
                for _ in range(len(clone.players[player_id].hand)):
                    card = clone.players[player_id].hand.pop()
                    clone.dealer.deck.append(card)
            clone.dealer.shuffle()
            clone.dealer.deck.insert(0, briscola_spy)
            for player_id in range(len(clone.players)):
                if player_id == clone.current_player_id:
                    continue
                for _ in range(hand_sizes[player_id]):
                    clone.players[player_id].hand.append(clone.dealer.deal())
            
        clone.render_mode = None
        return clone