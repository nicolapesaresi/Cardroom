import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.utils.scoring import play_n_games

class EndgameSolver:
    """Calculates the perfect strategy for an endgame."""
    def __init__(self, env:BriscolaEnv|None = None):
        if env is None:
            self.generate_endgame()
        else:
            state = env.get_state()
            # check that env in at the start of the endgame
            assert state["deck_left"] == 0, f"deck still has {state["deck_left"]} cards."
            assert len(state["cards_on_table"]) == 0, "there's card on table."
            assert len(state["hands"][0]) == len(state["hands"][1]) == 3, "players don't have 3 cards."
            self.env = env

    def generate_endgame(self) -> BriscolaEnv:
        """Generates and endgame env."""
        env = BriscolaEnv(names = ["p1", "p2"], render_mode=None)
        agents = [RandomAgent() for _ in range(env.n_players)]
        
        while env.get_state()["turn_counter"] < 18:
            player_id = env.current_player_id
            agent = agents[player_id]
            action = agent.select_action(env.get_observation())
            env.step(action)
        state = env.get_state()
        assert state["deck_left"] == 0, f"deck still has {state["deck_left"]} cards."
        assert len(state["cards_on_table"]) == 0, "there's card on table."
        assert len(state["hands"][0]) == len(state["hands"][1]) == 3, "players don't have 3 cards."
        self.env = env

    @staticmethod
    def state_key(state: dict) -> tuple:
        """
        Create a canonical, hashable key for a state (information set).
        This must include only *visible* information for the acting player.
        """
        # Hand: keep slot order (so actions map to slots). Represent card as (face_id, suit_id)
        hands = tuple(
            frozenset(Counter((c.suit_id, c.face_id) for c in state["hands"][i]).items())
            for i in range(len(state["hands"]))
        )
        # Cards on table: keep order they were played
        table = tuple((c.face_id, c.suit_id) for c in state.get("cards_on_table", []))
        # Include simple scalar info
        key = (
            hands,
            table,
            int(state.get("turn_counter", 0)),
            int(state.get("current_player", 0)),
            int(state.get("briscola_id", -1)),
            int(state.get("deck_left", 0)),
            tuple(state.get("points", (0,0))),
        )
        return key

    def solve_endgame(self):
        """
        Brute force evaluation of all endgame outcomes and calculates optimal policy.
        Returns:
            policy: dict mapping state keys to optimal action for current player
            results: list of all outcomes with actions, card names, turns, and final points
        """
        results = []
        policy = {}

        def minimax(env, actions=[], cards=[], turns=[]):
            state = env.get_state()

            if state["done"]:
                results.append({
                    "actions": actions,
                    "cards": cards,
                    "turns": turns,
                    "points": state["points"]
                })
                return state["points"]

            player_id = env.current_player_id
            best_score = None
            best_action = None

            for action in env.get_legal_actions():
                new_env = env.clone()
                # Get the card corresponding to this action
                card = str(state["hands"][player_id][action])
                new_env.step(action)
                scores = minimax(
                    new_env,
                    actions + [action],
                    cards + [card],
                    turns + [player_id]
                )

                if best_score is None or scores[player_id] > best_score[player_id]:
                    best_score = scores
                    best_action = action

            policy[self.state_key(env.get_state())] = best_action
            return best_score

        minimax(self.env.clone())
        return policy, results
    
    def plot_tree(self, policy, results):
        p0_color = "deepskyblue"
        p1_color = "lightblue"
        optimal_path_color = "limegreen"
        badmove_color = "coral"
        bestatnode_color = "gold"
        draw_color = "silver"

        fig, ax = plt.subplots(figsize=(25, 10))
        ax.set_axis_off()

        class Node:
            def __init__(self, env, parent=None, parent_action=None):
                self.env = env
                self.parent = parent
                self.parent_action = parent_action
                self.children = {}  # action -> Node
                self.x = 0
                self.y = 0
                self.width = 1  # subtree width

        root = Node(self.env.clone())

        # Build tree from sequences
        for res in results:
            node = root
            for action in res["actions"]:
                if action not in node.children:
                    new_env = node.env.clone()
                    new_env.step(action)
                    node.children[action] = Node(new_env, parent=node, parent_action=action)
                node = node.children[action]

        # Compute subtree widths
        def compute_width(node):
            if not node.children:
                node.width = 1
            else:
                node.width = sum(compute_width(child) for child in node.children.values())
            return node.width

        compute_width(root)

        # Assign positions
        def assign_positions(node, x_min=0, y=0):
            node.y = y
            if not node.children:
                node.x = x_min + node.width / 2
                return x_min + node.width
            curr_x = x_min
            for child in node.children.values():
                curr_x = assign_positions(child, curr_x, y - 1)
            children_x = [c.x for c in node.children.values()]
            node.x = sum(children_x) / len(children_x)
            return x_min + node.width

        assign_positions(root)

        # Prepare alternating offsets for edge labels
        edges_by_depth = {}
        def collect_edges(node):
            for child in node.children.values():
                depth = int(-child.y)
                edges_by_depth.setdefault(depth, []).append(child)
                collect_edges(child)
        collect_edges(root)

        edge_offsets = {}
        for depth, nodes_at_depth in edges_by_depth.items():
            nodes_sorted = sorted(nodes_at_depth, key=lambda n: n.x)
            for i, node in enumerate(nodes_sorted):
                edge_offsets[node] = 0.1 * (1 if i % 2 == 0 else -1)

        # --- Identify the single optimal path from root to leaf ---
        optimal_path_nodes = set()
        env_clone = root.env.clone()
        node = root
        while not env_clone.get_state()["done"]:
            key = self.state_key(env_clone.get_state())
            act = policy.get(key)
            if act is None:
                break
            # Add the child node along this action
            child = node.children.get(act)
            if child is None:
                break
            optimal_path_nodes.add(child)
            env_clone.step(act)
            node = child

        # Plot recursively
        def plot_node(node):
            state = node.env.get_state()
            x, y = node.x, node.y

            # Node label
            if state["done"]:
                points = state["points"]
                label = f"P0: {points[0]}\nP1: {points[1]}"
                color = p0_color if getattr(node.env, "result", 0) == 1 else \
                        p1_color if getattr(node.env, "result", 0) == -1 else draw_color
            else:
                label = f"P{state['current_player']}"
                color = p0_color if state["current_player"] == 0 else p1_color

            ax.text(x, y, label, ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor=color, edgecolor= "black"))

            # Edge from parent
            if node.parent is not None:
                parent_x, parent_y = node.parent.x, node.parent.y
                parent_key = self.state_key(node.parent.env.get_state())
                optimal_action = policy.get(parent_key, None)
                # Default green for edges along optimal policy
                highlight = (node.parent_action == optimal_action)
                color_line = bestatnode_color if highlight else badmove_color
                # Override to lime if this edge is along the single optimal path
                if node in optimal_path_nodes:
                    color_line = optimal_path_color

                ax.plot([parent_x, x], [parent_y, y],
                        color=color_line,
                        linewidth=2.5 if node in optimal_path_nodes else 1.0)

                # Card name and turn
                current_player = node.parent.env.get_state()["current_player"]
                card = str(node.parent.env.get_state()["hands"][current_player][node.parent_action])
                edge_label = f"[{node.parent_action}] {card.replace(' of', '')}"

                # Use alternating offset
                offset_y = edge_offsets.get(node, 0)
                ax.text((parent_x + x) / 2, (parent_y + y) / 2 + offset_y,
                        edge_label, ha="center", va="center", fontsize=8)

            for child in node.children.values():
                plot_node(child)

        plot_node(root)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=optimal_path_color, lw=2.5, label='Optimal strategy'),
            Line2D([0], [0], color=bestatnode_color, lw=2.5, label='Correct action'),
            Line2D([0], [0], color=badmove_color, lw=1, label='Bad action'),
            Line2D([0], [0], marker='o', color='w', label='P0 move', markerfacecolor=p0_color, markersize=12, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='P1 move', markerfacecolor=p1_color, markersize=12, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Draw', markerfacecolor=draw_color, markersize=12, markeredgecolor='k'),
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                ncol=6, fontsize=10)

        # Title with predicted winner along optimal play
        points = env_clone.get_state()['points']
        winner = "P0" if points[0] > points[1] else "P1" if points[1] > points[0] else "Draw"
        ax.set_title(f"Endgame Tree [Optimal Winner: {winner}]", fontsize=16)

        plt.show()












    # def brute_force(self):
    #     """
    #     Brute force evaluation of all endgame outcomes.
    #     Returns:
    #         results: list of dicts with action sequence and final scores.
    #     """
    #     stack = [(self.env.clone(), [])]  # env copy + sequence
    #     results = []

    #     while stack:
    #         env, sequence = stack.pop()
    #         state = env.get_state()

    #         # End of game
    #         if state["done"]:
    #             results.append({
    #                 "sequence": sequence,
    #                 "scores": env.result
    #             })
    #             continue

    #         # Expand children
    #         player_id = env.current_player_id
    #         for action in env.get_legal_actions():
    #             new_env = env.clone()
    #             new_env.step(action)
    #             stack.append((new_env, sequence + [(player_id, action)]))

    #     return results



class EndgameEvaluator:
    """Evaluates agent ability in the endgame (the last three cards), which has perfect information."""
    def __init__(self, agent: Agent, ):
        self.agent = agent
    
