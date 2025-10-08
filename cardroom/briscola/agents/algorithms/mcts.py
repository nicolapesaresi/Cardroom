import math
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Any, List, Optional

from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.bot import BotAgent  # optional rollout policy


def obs_to_key(obs: dict) -> Tuple:
    """
    Create a canonical, hashable key for an observation (information set).
    This must include only *visible* information for the acting player.
    """
    # Hand: keep slot order (so actions map to slots). Represent card as (face_id, suit_id)
    hand = frozenset(Counter((c.suit_id, c.face_id) for c in obs["hand"]).items())
    # Cards on table: keep order they were played
    table = tuple((c.face_id, c.suit_id) for c in obs.get("cards_on_table", []))
    # Include simple scalar info
    key = (
        hand,
        table,
        int(obs.get("turn_counter", 0)),
        int(obs.get("current_player", 0)),
        int(obs.get("briscola_id", -1)),
        int(obs.get("deck_left", 0)),
        int(obs.get("current_player_points", 0)),
        int(obs.get("other_player_points", 0)),
    )
    return key


class InfoSetNode:
    """Node keyed by information set (observation). Stats are aggregated per action."""

    def __init__(self, obs_key: Tuple, obs: dict):
        self.obs_key = obs_key
        # who is the player to move at this information set
        self.player: int = int(obs["current_player"])
        # actions are indices of the hand slots (0 .. len(hand)-1)
        self.legal_actions: List[int] = list(range(len(obs["hand"])))
        # stats per action
        self.N: Dict[int, int] = {a: 0 for a in self.legal_actions}    # visit counts
        self.W: Dict[int, float] = {a: 0.0 for a in self.legal_actions}  # total value (from this node.player perspective)
        self.Q: Dict[int, float] = {a: 0.0 for a in self.legal_actions}  # mean value = W/N
        self.children: Dict[int, Tuple] = {}  # action -> child_obs_key
        self.total_visits: int = 0

    def update_stats(self, action: int, value: float) -> None:
        """Update stats for a chosen action with a value expressed from this node's player's perspective."""
        self.N[action] += 1
        self.W[action] += value
        self.Q[action] = self.W[action] / self.N[action]
        self.total_visits += 1


class ISMCTS:
    """
    Information-Set MCTS (merge nodes with identical observations).
    - sample new determinization at root for each simulation (simple PIMC+IS merging)
    - do selection/expansion using a determinized env
    - rollout with provided policy (random or BotAgent)
    """

    def __init__(self, root_env: BriscolaEnv, cpuct: float = 1.4, rollout_policy: str = "random"):
        self.root_env = root_env
        root_obs = self.root_env.get_observation()
        self.root_key = obs_to_key(root_obs)
        self.info_map: Dict[Tuple, InfoSetNode] = {}
        self.cpuct = cpuct
        self.rollout_policy = rollout_policy

        # create root node
        self._get_node(self.root_key, root_obs)

    def _get_node(self, key: Tuple, obs: Optional[dict] = None) -> InfoSetNode:
        """Return node for key; create it using obs if it doesn't exist."""
        if key in self.info_map:
            return self.info_map[key]
        if obs is None:
            raise ValueError("Must provide obs to create new node")
        node = InfoSetNode(key, obs)
        self.info_map[key] = node
        return node

    def select_action(self, simulations: int = 200) -> int:
        """Run many simulations and return best action (by visit count) from root."""
        for _ in range(simulations):
            env_det = self.root_env.clone_from_observation()  # sample determinization
            self._run_simulation(env_det)

        root_node = self.info_map[self.root_key]
        # choose action with max visits (tie-break randomly)
        best_action = max(root_node.legal_actions, key=lambda a: (root_node.N.get(a, 0), -np.random.random()))
        return int(best_action)

    def _run_simulation(self, env_det: BriscolaEnv) -> None:
        """
        Run a single simulation using determinized env_det.
        Build path of (node, action) pairs. Expand one new child when possible.
        Then rollout to terminal (or use a value estimate) and backpropagate.
        """
        path: List[Tuple[InfoSetNode, int]] = []

        # start from root (but use the determinized env's observation; root_key might map to same obs_key)
        obs = env_det.get_observation()
        key = obs_to_key(obs)
        node = self._get_node(key, obs)

        # Selection & Expansion loop
        while True:
            # If there exists any untried action (N[action] == 0) expand one
            untried = [a for a in node.legal_actions if node.N.get(a, 0) == 0]
            if untried:
                action = int(np.random.choice(untried))
                # step determinized env with that action
                env_det.step(action)
                next_obs = env_det.get_observation()
                next_key = obs_to_key(next_obs)
                child_node = self._get_node(next_key, next_obs)
                # register child mapping for this action
                node.children[action] = next_key
                path.append((node, action))
                node = child_node
                break  # we expanded a new action; now we will rollout from env_det
            # otherwise select action by UCT (Q + u)
            best_score = -float("inf")
            best_action = None
            total_N = node.total_visits + 1e-12
            for a in node.legal_actions:
                Q = node.Q.get(a, 0.0)
                N_a = node.N.get(a, 0)
                u = self.cpuct * math.sqrt(total_N) / (1 + N_a)
                score = Q + u
                if score > best_score:
                    best_score = score
                    best_action = a
            if best_action is None:
                # no legal action? terminal?
                break
            # step determinized env and move to child node
            env_det.step(int(best_action))
            next_obs = env_det.get_observation()
            next_key = obs_to_key(next_obs)
            child_node = self._get_node(next_key, next_obs)
            # ensure child mapping exists
            node.children[int(best_action)] = next_key
            path.append((node, int(best_action)))
            node = child_node
            # if env is terminal, stop selection
            if env_det.done:
                break

        # Leaf node is `node`. At this point env_det reflects the game after the expansion action.
        # Capture the leaf player: the player to move at the leaf before rollout.
        # current_player is the player to move in the observed state.
        leaf_player = env_det.current_player_id

        # Evaluate leaf: rollout to terminal from env_det (determinized) using chosen policy
        v_leaf = self._rollout_value(env_det, leaf_player)

        # Backpropagate: update each node/action on the path.
        # Important: v_leaf is expressed from leaf_player's perspective.
        # For a node whose acting player == leaf_player, the value is v_leaf;
        # otherwise it is -v_leaf. This handles cases where the same player may act consecutively.
        for (node_obj, action_taken) in reversed(path):
            node_player = node_obj.player
            # value from node player's perspective
            value_for_node = v_leaf if node_player == leaf_player else -v_leaf
            node_obj.update_stats(action_taken, value_for_node)

    def _rollout_value(self, env_det: BriscolaEnv, leaf_player: int) -> float:
        """
        Run a rollout policy to terminal and return value in {-1, 0, 1} from the leaf_player's perspective.
        - If rollout_policy == "random": play random legal moves.
        - If rollout_policy == "bot": use BotAgent to choose actions (may require env.get_observation() signature).
        """
        # if already terminal, compute result immediately
        if env_det.done:
            env_result, _ = env_det.check_result()
            return self._result_to_value(env_result, leaf_player)

        if self.rollout_policy == "bot":
            bot = BotAgent()
            while not env_det.done:
                obs = env_det.get_observation()
                legal = list(range(len(obs["hand"])))
                if not legal:
                    break
                action = bot.select_action(obs)
                # safety: if bot returns None or invalid, fall back to random
                if action is None or int(action) not in legal:
                    action = int(np.random.choice(legal))
                env_det.step(int(action))
        else:
            # random rollout
            while not env_det.done:
                obs = env_det.get_observation()
                legal = list(range(len(obs["hand"])))
                if not legal:
                    break
                action = int(np.random.choice(legal))
                env_det.step(action)

        env_result, _ = env_det.check_result()
        return self._result_to_value(env_result, leaf_player)

    @staticmethod
    def _result_to_value(env_result: int, perspective_player: int) -> float:
        """
        Convert env_result (0 draw, 1 p0 win, -1 p1 win) to value from perspective_player (0 or 1).
        Returns +1 if perspective_player wins, -1 if loses, 0 if draw.
        """
        if env_result == 0:
            return 0.0
        if env_result == 1:
            winner = 0
        else:
            winner = 1
        return 1.0 if winner == perspective_player else -1.0
