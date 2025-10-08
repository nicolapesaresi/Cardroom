import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import seaborn as sns
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from cardroom.briscola.game.cards import SUITS
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.game.player import BriscolaPlayer
from cardroom.briscola.agents.agent import Agent
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.agents.human import HumanAgent
from cardroom.briscola.agents.bot import BotAgent
from cardroom.briscola.agents.superbot import SuperbotAgent
from cardroom.briscola.agents.optimus import OptimusAgent
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.utils.scoring import play_n_games

class EndgameSolver:
    """Calculates the optimal strategy for an endgame."""
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
        agents = [BotAgent() for _ in range(env.n_players)] # generate with Bot so endgames are less absurd
        
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

        return env

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
            policy: dict mapping state keys to a *set* of optimal actions for current player
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
            best_actions = set()
            scores_for_actions = {}

            for action in env.get_legal_actions():
                new_env = env.clone()
                card = str(state["hands"][player_id][action])
                new_env.step(action)

                scores = minimax(
                    new_env,
                    actions + [action],
                    cards + [card],
                    turns + [player_id]
                )
                scores_for_actions[action] = scores

                if best_score is None or scores[player_id] > best_score[player_id]:
                    best_score = scores
                    best_actions = {action}
                elif scores[player_id] == best_score[player_id]:
                    best_actions.add(action)

            # Store all equally-optimal actions
            policy[self.state_key(env.get_state())] = best_actions
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

        # --- Identify all optimal paths (subtree) ---
        optimal_path_nodes = set()

        def collect_optimal_subtree(node):
            state = node.env.get_state()
            if state["done"]:
                return
            key = self.state_key(state)
            optimal_actions = policy.get(key, set())
            for act, child in node.children.items():
                if act in optimal_actions:
                    optimal_path_nodes.add(child)
                    collect_optimal_subtree(child)

        collect_optimal_subtree(root)

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
                optimal_actions = policy.get(parent_key, set())

                # Highlight if this edge corresponds to one of the optimal moves
                highlight = (node.parent_action in optimal_actions)
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

        # All optimal paths lead to the same points, so pick the first result that follows the optimal policy
        optimal_final_points = None
        for res in results:
            node_env = self.env.clone()
            follows_optimal = True
            for i, action in enumerate(res["actions"]):
                key = self.state_key(node_env.get_state())
                if action not in policy.get(key, set()):
                    follows_optimal = False
                    break
                node_env.step(action)
            if follows_optimal:
                optimal_final_points = res["points"]
                break

        if optimal_final_points is None:
            optimal_final_points = results[0]["points"] if results else (0, 0)

        winner_optimal = (
            "P0" if optimal_final_points[0] > optimal_final_points[1] else
            "P1" if optimal_final_points[1] > optimal_final_points[0] else
            "Draw"
        )

        ax.set_title(
            f"Endgame Tree [Optimal Outcome: P0 {optimal_final_points[0]} - P1 {optimal_final_points[1]} → Winner: {winner_optimal}] "
            f"Briscola: {SUITS[self.env.briscola_suit_id]}",
            fontsize=16
        )
        plt.show()


class EndgameEvaluator:
    """Evaluates an agent's play in Briscola endgames against the perfect strategy."""
    def __init__(self, agent: Agent):
        self.agent = agent

    def evaluate_endgame(self, endgame: BriscolaEnv | None = None) -> dict:
        """
        Evaluate a single endgame against a perfect opponent.
        Metrics:
            - accuracy: fraction of agent moves that are optimal
            - regret_points: points lost due to suboptimal moves
            - regret_result: 1 if the game result would change if agent played optimally
            - final_points: actual game points
            - winner: actual game winner
            - optimal_final_points: points if agent played perfectly
            - played_actions: list of actions agent actually took
        """
        solver = EndgameSolver(endgame)
        policy, results = solver.solve_endgame()

        env_agent = solver.env.clone()
        total_decisions = 0
        optimal_decisions = 0
        played_actions = []

        # Play game with agent against perfect opponent
        while not env_agent.get_state()["done"]:
            state = env_agent.get_state()
            player_id = state["current_player"]
            key = solver.state_key(state)
            optimal_actions = list(policy.get(key, set()))

            if player_id == 0:  # our agent
                obs = env_agent.get_observation()
                if isinstance(self.agent, DonatelloAgent):
                    action = self.agent.select_action(obs, env_agent.clone_from_observation())
                elif isinstance(self.agent, SuperbotAgent) or isinstance(self.agent, OptimusAgent):
                    action = self.agent.select_action(obs, env_agent.clone())
                else:
                    action = self.agent.select_action(obs)

                played_actions.append(action)
                if optimal_actions:
                    total_decisions += 1
                    if action in optimal_actions:
                        optimal_decisions += 1

            else:  # opponent plays perfectly
                action = next(iter(optimal_actions)) if optimal_actions else None

            env_agent.step(action)

        final_points = env_agent.get_state()["points"]
        winner = "Agent(P0)" if final_points[0] > final_points[1] else \
                 "Opponent(P1)" if final_points[1] > final_points[0] else "Draw"
        accuracy = optimal_decisions / total_decisions if total_decisions else 0.0

        # Simulate perfect play from root for optimal points
        env_optimal = solver.env.clone()
        while not env_optimal.get_state()["done"]:
            state = env_optimal.get_state()
            key = solver.state_key(state)
            optimal_actions = list(policy.get(key, set()))
            # pick any optimal action
            action = next(iter(optimal_actions)) if optimal_actions else None
            env_optimal.step(action)
        optimal_final_points = env_optimal.get_state()["points"]

        regret_points = optimal_final_points[0] - final_points[0]
        regret_result = int((final_points[0] > final_points[1]) != 
                            (optimal_final_points[0] > optimal_final_points[1]))

        return {
            "total_decisions": total_decisions,
            "optimal_decisions": optimal_decisions,
            "accuracy": accuracy,
            "regret_points": regret_points,
            "regret_result": regret_result,
            "final_points": final_points,
            "optimal_final_points": optimal_final_points,
            "played_actions": played_actions
        }

    def evaluate_agent(self, n_endgames: int=500) -> dict:
        """Evaluates an agent on many endgames and aggregate and plot the stats."""
        results = []
        for _ in tqdm(range(n_endgames)):
            results.append(self.evaluate_endgame())
        fig = self.plot_results(results)
        return results, fig

    @staticmethod
    def plot_results(results: list[dict]):
        """Generate summary plots for a list of EndgameEvaluator results in a single figure."""
        # Extract metrics
        accuracies = [r["accuracy"] for r in results]
        regret_points = [r["regret_points"] for r in results]
        regret_result = [r["regret_result"] for r in results]
        final_points = [r["final_points"][0] for r in results]
        optimal_points = [r["optimal_final_points"][0] for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()

        # Accuracy count plot (discrete)
        df_acc = pd.DataFrame({"Accuracy": [round(a, 2) for a in accuracies]})
        sns.countplot(data=df_acc, x="Accuracy", ax=axes[0], color="lime")
        axes[0].set_title("Accuracy Across Games")
        axes[0].set_xlabel("Accuracy (fraction of optimal moves)")
        axes[0].set_ylabel("Number of games")

        # Regret points histogram
        sns.histplot(regret_points, bins=20, kde=True, ax=axes[1], color="red")
        axes[1].set_title("Total Points Lost (Regret) Distribution")
        axes[1].set_xlabel("Points lost due to suboptimal play")
        axes[1].set_ylabel("Number of games")

        # Result regret count plot with labels Yes/No
        counts = np.bincount(regret_result, minlength=2)
        df_counts = pd.DataFrame({
            "Result Regret": ["No", "Yes"],
            "Count": counts
        })

        sns.barplot(
            data=df_counts,
            x="Result Regret",
            y="Count",
            hue="Result Regret",
            palette="bright",
            dodge=False,
            ax=axes[2],
            legend=False
        )
        axes[2].set_title("Number of Games with Result Regret")
        axes[2].set_xlabel("Result would have changed")
        axes[2].set_ylabel("Number of games")
        for i, row in df_counts.iterrows():
            axes[2].text(i, row["Count"] + 0.05, str(row["Count"]), ha='center', va='bottom')

        # Final points vs Optimal points scatter
        axes[3].scatter(optimal_points, final_points, color='orange', alpha=0.6)
        max_val = max(max(optimal_points), max(final_points))
        axes[3].plot([0, max_val], [0, max_val], 'r--', label="Optimal")
        axes[3].set_title("Agent Final Points vs Optimal Points")
        axes[3].set_xlabel("Optimal Points")
        axes[3].set_ylabel("Agent Points")
        axes[3].legend()

        plt.tight_layout()
        plt.show()
        return fig