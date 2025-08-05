import yaml
from cardroom.scopa.game.dealer import ScopaDealer
from cardroom.scopa.game.player import ScopaPlayer
from cardroom.scopa.agents.random import RandomAgent

RENDER_MODES = [None, "text"]
RULES_YAML = "cardroom\scopa\game\rules.yaml"

class ScopaEnv:
    """Environment for a game of Briscola."""
    def __init__(self, variant: str="liscia", players: list[ScopaPlayer,ScopaPlayer]|None = None, render_mode: str = "text"):
        """Instantiates the environment.
        Args:
            variant: variant of the game.
            players: list of players who are going to play the game.
            render_mode: render mode for the game.
        """
        self.n_players = 2
        if render_mode not in RENDER_MODES:
            raise NotImplementedError(f"Render mode {render_mode} not implented.")
        self.render_mode = render_mode
        self.variant = variant
        self.get_rules(variant)
        self.instantiate_players(players)
        self.reset()
        self.render()

    def get_rules(self, variant):
        """Retrieves rules configuration from yaml.
        Args:
            variant: variant of the game, see the yaml for supported variants.
        """
        with open(RULES_YAML, "r") as f:
            all_rules = yaml.safe_load(f)
        self.rules = all_rules["variants"].get(variant, None)
        if self.rules is None:
            raise KeyError(f"Got {variant} variant, expected one of {all_rules['variants'].keys()}")


    def instantiate_players(self, players: list[ScopaPlayer, ScopaPlayer] | None = None):
        """Instantiates the players for the game.
        Args:
            players: list of players who are going to play the game.
        """
        # if players has not been provided, instantiate two random agents
        if players is None:
            self.players = [ScopaPlayer(agent=RandomAgent(name=f"RandomP{i}")) for i in range(self.n_players)]
        # if players has been provided, check that it is a valid format
        else:
            if len(players) != self.n_players:
                raise ValueError(f"Expected {self.n_players} players, got {len(players)}")
            for player in players:
                if not isinstance(player, ScopaPlayer):
                    raise TypeError(f"Expected ScopaPlayer, got {type(player)}")
            self.players = players

    #TODO: UPDATE FROM HERE ON:

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
            if card.suit_id == first_suit_id:
                card.rank += 50
            played_rank.append(card.rank)
            played_points += card.points
        winning_card_idx = played_rank.index(max(played_rank))
        winning_player_id = self.turn_order[winning_card_idx]
        self.players[winning_player_id].points += played_points
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
        state["current_player"] = self.current_player_id
        state["turn_order"] = self.turn_order
        state["turn_counter"] = self.turn_counter
        state["cards_on_table"] = self.cards_on_table
        state["briscola_id"] = self.briscola_suit_id
        state["briscola_spy"] = self.briscola_spy
        state["hands"] = [player.hand for player in self.players]
        state["points"] = [player.points for player in self.players]
        state["all_played_cards"] = self.played_cards_history
        state["turn_history"] = self.turn_history
        state["done"] = self.done

        return state
    
    def get_observation(self) -> dict:
        """Retrieves observation of game state for current player.
        Returns:
            obs: observation of the game state for the current player.
        """
        state = self.get_state()
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
                self.done = True
            else:
                if self.dealer.cards_left > 0:
                    for id in self.turn_order:
                        player = self.players[id]
                        player.hand.append(self.dealer.deal())
        self.render()

    def render(self):
        """Renders the current state of the game."""
        state = self.get_state()

        if self.render_mode is None:
            return # don't render
        elif self.render_mode == "text":
            self.text_render(state)

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