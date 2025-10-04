import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
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
from cardroom.briscola.utils.scoring import play_flipped_games
from cardroom.briscola.agents.evaluation.evaluate_endgame import EndgameEvaluator
from scipy.stats import binomtest


class GameEvaluator:
    """Evaluates a single finished game by examining briscolas and carichi each player played."""

    def __init__(self, state: dict):
        """
        Args:
            state: final state of a finished game.
        """
        if not state.get("done", False):
            raise ValueError("state corresponds to a game not yet finished.")
        self.state = state

    @staticmethod
    def deck_analysis(state: dict) -> dict:
        """
        Analyze briscola and carichi played by each player from a final state.

        Returns:
            stats: dict keyed by player name with
                - n_briscolas
                - avg_briscola_rank
                - briscola_points
                - avg_briscola_value
                - n_carichi
                - avg_carico_value
        """
        all_played_cards = np.array(state["all_played_cards"])
        turn_history = np.array(state["turn_history"])
        trick_winners = np.array(state["trick_winners"])
        player_names = state["player_names"]
        n_players = len(player_names)

        stats = {}
        n_tricks = len(trick_winners)

        for pid, name in enumerate(player_names):
            briscolas, briscola_values = [], []
            carichi, carico_values = [], []

            for trick_idx in range(n_tricks):
                # cards and players in this trick
                start = trick_idx * n_players
                end = (trick_idx + 1) * n_players
                trick_cards = all_played_cards[start:end]
                trick_players = turn_history[start:end]
                winner_pid = trick_winners[trick_idx]

                # card played by this pid in this trick
                player_card_idx = np.where(trick_players == pid)[0][0]
                player_card = trick_cards[player_card_idx]

                trick_points = sum(c.points for c in trick_cards)
                sign = 1 if winner_pid == pid else -1

                # briscola
                if player_card.is_briscola:
                    briscolas.append(player_card)
                    briscola_values.append(sign * trick_points)

                # carichi: Ace (1) and Three (3) that are not briscola
                if (not player_card.is_briscola) and (player_card.face_id in [1, 3]):
                    carichi.append(player_card)
                    carico_values.append(sign * trick_points)

            n_briscolas = len(briscolas)
            mean_rank = (np.mean([c.rank for c in briscolas]) - 100) if briscolas else 0
            briscola_points = sum(c.points for c in briscolas)
            avg_briscola_value = float(np.mean(briscola_values)) if briscola_values else 0.0

            n_carichi = len(carichi)
            avg_carico_value = float(np.mean(carico_values)) if carico_values else 0.0

            stats[name] = {
                "n_briscolas": int(n_briscolas),
                "avg_briscola_rank": float(mean_rank),
                "briscola_points": int(briscola_points),
                "avg_briscola_value": float(avg_briscola_value),
                "n_carichi": int(n_carichi),
                "avg_carico_value": float(avg_carico_value),
            }

        return stats

    def from_state(self) -> dict:
        """Instance wrapper calling the static deck_analysis on this object's state."""
        return GameEvaluator.deck_analysis(self.state)


class DeckMetricsEvaluator:
    """Analyzes many games using GameEvaluator and returns player and global stats."""

    def __init__(self, game_states:list[dict]):
        self.game_states = game_states
        self.stats = []

    def analyze_games(self):
        """Analyze ganes via GameEvaluator and store the result."""
        for game_state in self.game_states:
            stats = GameEvaluator.deck_analysis(game_state)
            self.stats.append(stats)

    def calculate_metrics(self) -> pd.DataFrame:
        """Compute metrics per player across all games."""
        records = []
        for game_stats in self.stats:
            for player, stats in game_stats.items():
                record = {"player": player}
                record.update(stats)
                records.append(record)

        if not records:
            return pd.DataFrame(columns=[
                "player", "n_briscolas", "avg_briscola_rank", "briscola_points",
                "avg_briscola_value", "n_carichi", "avg_carico_value"
            ])

        df = pd.DataFrame(records)
        # mean across games per player for the provided metrics
        df_agg = df.groupby("player", as_index=False).mean(numeric_only=True)
        return df_agg

    @staticmethod
    def plot_metrics(df_metrics: pd.DataFrame):
        """Plots one subplot per metric, one bar per player on each subplot, with value labels."""
        if df_metrics.empty:
            raise ValueError("df_metrics is empty. Run per_player_metrics first.")

        players = df_metrics["player"].tolist()
        x = np.arange(len(players))

        metrics = [
            ("n_briscolas", "N° Briscolas", "yellow", 5),
            ("avg_briscola_rank",  "Avg Briscola Rank",  "lightyellow", 5),
            ("briscola_points",    "Avg Points of Briscolas", "gold", 3.1),
            ("avg_briscola_value", "Avg Briscola Value", "orange", None),
            ("n_carichi",          "N° Carichi",          "royalblue", 3),
            ("avg_carico_value",   "Avg Carico Value",   "cornflowerblue", None),
        ]

        n_rows, n_cols = 2, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.6 * n_rows), constrained_layout=True)
        axes_flat = np.ravel(axes)

        # Plot each metric on its own axis
        for ax, (col, title, color, mean_val) in zip(axes_flat, metrics):
            if col not in df_metrics.columns:
                ax.axis("off")
                continue

            vals = df_metrics[col].to_numpy()
            bars = ax.bar(x, vals, color=color, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(players, rotation=15)
            ax.set_title(title)
            ax.set_ylabel("Value")
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            if mean_val is not None:
                ax.axhline(mean_val, color="black", linestyle="--", linewidth=1, alpha=0.8)

            try:
                ax.bar_label(bars, fmt="%.2f", padding=2)
            except Exception:
                ymax = np.nanmax(vals) if len(vals) and np.isfinite(np.nanmax(vals)) else 1.0
                offset = 0.02 * (ymax if ymax != 0 else 1.0)
                for rect, v in zip(bars, vals):
                    if not np.isfinite(v):
                        continue
                    ax.text(rect.get_x() + rect.get_width()/2, v + offset,
                            f"{v:.2f}" if isinstance(v, float) else f"{int(v)}",
                            ha="center", va="bottom", fontsize=9, color="black")

        # Turn off any remaining unused axes (if metrics < n_rows*n_cols)
        for ax in axes_flat[len(metrics):]:
            ax.axis("off")

        plt.show()
        return fig
    
    def run(self, plot = True):
        """Runs the pipeline."""
        self.analyze_games()
        metrics = self.calculate_metrics()

        if plot:
            fig = self.plot_metrics(metrics)
            return metrics, fig
        return metrics

