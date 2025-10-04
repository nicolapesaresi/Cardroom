import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.cards import Card

def one_hot_cards(cards: list[Card], value:float=1.0, one_hot:np.ndarray|None = None):
    """Returns one hot encoding of the cards received in the list.
    Args:
        cards: list of cards to encode.
        values: value to encode for the passed cards. default 1.
        one_hot: optional one_hot encoded vector, passed if you don't want to start anew.
    Returns:
        one_hot: one_hot encoded array of cards.
    """
    if one_hot is None:
        one_hot = np.zeros(40)
    else:
        assert len(one_hot) == 40, "One hot encoded vector is not of length 40."

    for card in cards:
        card_id = card.card_id
        idx = card_id - 1 # index of the card in the one-hot encoded array

        one_hot[idx] = value
    return one_hot

def pad_array(x: np.ndarray, L: int, pad_val: float, dtype=None):
    """Pad/truncate 1D array x to length L with pad_val."""
    x = np.asarray(x, dtype=dtype if dtype is not None else float)
    y = np.full(L, pad_val, dtype=x.dtype)
    n = min(L, x.shape[0])
    y[:n] = x[:n]
    return y

def encode_state(state: dict) -> np.ndarray:
    """Encodes the state into an input for a neural network.
    Args:
        state: state of the game as returned from env.get_observation()
    Returns:
        encoded state
    """
    current_player_id = state["current_player"]
    turn_counter = state["turn_counter"] / 20.0

    cards_on_table = one_hot_cards(state["cards_on_table"])

    briscola_id = np.zeros(4)
    briscola_id[state["briscola_id"]] = 1.0
    briscola_spy = one_hot_cards([state["briscola_spy"]])
    
    played_ids = []
    for card in state["all_played_cards"]:
        idx = card.card_id - 1  # 0..39
        played_ids.append(idx / 39.0)
    all_played_cards = pad_array(np.array(played_ids, dtype=np.float32), 39, pad_val=-1.0, dtype=np.float32)

    turn_hist = np.array(state["turn_history"], dtype=np.int8)
    turn_history = pad_array(turn_hist, 39, pad_val=-1, dtype=np.int8).astype(np.float32)

    hand = one_hot_cards(state["hand"])
    gone_cards = one_hot_cards(state["all_played_cards"])
    gone_cards = one_hot_cards(state["hand"], one_hot=gone_cards)
    current_player_points = state["current_player_points"] / 120
    other_player_points = state["other_player_points"] / 120

    encoded_state = np.concatenate([
        np.array([current_player_id], dtype=np.float32), # 1
        np.array([turn_counter], dtype=np.float32),  # 1
        briscola_id,                        # 4
        cards_on_table,                     # 40
        briscola_spy,                       # 40
        hand,                               # 40
        gone_cards,                         # 40
        np.array([current_player_points, other_player_points], dtype=np.float32),  # 2
        all_played_cards,                   # 39 (padded with -1)
        turn_history,                       # 39 (padded with -1)
    ], dtype=np.float32)

    return encoded_state # final length: 246

def opponent_hand_output(state: dict, player_id: int) -> np.ndarray:
    """Returns the correct output for the belief state net for the state (not obs) of the game.
    This is a 40-length one-hot encoded array with 1s for the cards in the opponent hand.
    Args:
        state: state of the game, as returned by env.get_state().
        player_id: id of the opponent.
    Returns:
        output: one-hot encoded vector of the opponent's hand.
    """
    hand = state["hands"][player_id]
    output = one_hot_cards(hand)
    return output
