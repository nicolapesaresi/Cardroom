import math
import numpy as np
from typing import Self
from collections import defaultdict
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.bot import BotAgent

def get_legal_actions(observation: dict) -> list:
    return list(range(len(observation["hand"])))

class MCTSNode:
    def __init__(self, observation: dict, env: BriscolaEnv, parent: Self=None, parent_action: int=None):
        self.observation = observation
        self.env = env
        self.parent = parent
        self.parent_action = parent_action
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.possible_actions = get_legal_actions(observation)

    def expand(self) -> Self:
        """Expand a single action."""
        child_env = self.env.clone_from_observation()
        action = np.random.choice(self.possible_actions)
        child_env.step(int(action))
        child_observation = child_env.get_observation()
        child = MCTSNode(child_observation, child_env, parent=self, parent_action=action)
        self.children.append(child)
        return child

    def is_terminal(self):
        return self.env.done
    
    def random_action_policy(self):
        legal_actions = get_legal_actions(self.observation)
        return np.random.choice(legal_actions) if legal_actions else None
    
    def bot_action_policy(self):
        bot = BotAgent()
        action = bot.select_action(self.observation)
        return action

    def rollout(self):
        env = self.env.clone_from_observation()
        current_player_id = env.current_player_id
        while not env.done:
            legal_actions = get_legal_actions(env.get_observation())
            if not legal_actions:
                break
            # action = np.random.choice(legal_actions)
            action = self.bot_action_policy()
            env.step(int(action))
        env_result, _ = env.check_result()
        if env_result == 0:
            return 0
        elif (env_result == 1 and current_player_id == 0) or (env_result == -1 and current_player_id == 1):
            return 1
        else:
            return -1
    
    def backpropagate(self, result):
        self.visits += 1
        if result == 1:
            self.wins += 1
        elif result == 0:
            self.draws += 1
        else:
            self.losses += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.children) == len(self.possible_actions)
    
    def best_child(self, exploration=0.1):
        child_stats: dict[int, dict[str, float]] = {}
        for child in self.children:
            a = child.parent_action
            if a not in child_stats:
                child_stats[a] = {
                    "wins": child.wins,
                    "draws": child.draws,
                    "losses": child.losses,
                    "visits": child.visits,
                }
            else:
                child_stats[a]["wins"]   += child.wins
                child_stats[a]["draws"]  += child.draws
                child_stats[a]["losses"] += child.losses
                child_stats[a]["visits"] += child.visits

        weights = []
        for action in sorted(child_stats.keys()):
            stats = child_stats[action]
            exploitation = (stats["wins"] - stats["losses"]) / stats["visits"]
            exploration_term = exploration * np.sqrt(2 * np.log(self.visits) / stats["visits"])
            weight = exploitation + exploration_term
            weights.append(weight)

        best_action = sorted(child_stats.keys())[int(np.argmax(weights))]
        for child in self.children:
            if child.parent_action == best_action:
                return child

    def _tree_policy(self):
        """Keep going down recursively, expanding as needed."""
        current_node = self
        while not current_node.is_terminal():
            current_node = current_node.expand()
        return current_node

    def best_action(self, simulations):
        for _ in range(simulations):
            leaf = self._tree_policy()
            reward = leaf.rollout()
            leaf.backpropagate(reward)
        return int(self.best_child(exploration=0.0).parent_action)