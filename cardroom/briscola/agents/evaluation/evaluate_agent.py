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
from cardroom.briscola.utils.scoring import play_n_games

class EvaluateAgent:
    def __init__(self, agent: Agent, n_games: int, opponents: list[Agent] | None = None):
        self.agent = agent
        self.n_games = n_games
        if opponents is None:
            copy_agent = deepcopy(agent)
            copy_agent.name = agent.name + "_clone"
            self.opponents = [
                RandomAgent(),
                BotAgent(),
                DonatelloAgent(name="MCTS_randompolicy_200sim"),
                DonatelloAgent(name="MCTS_botpolicy_200sim"),
                copy_agent
            ]
        else:
            self.opponents = opponents
        self.results = pd.DataFrame(columns=["opponent", "winner", "game_index"])

    def play_matches(self) -> tuple[list[str], list[dict]]:
        all_results = []
        for opp in tqdm(self.opponents):
            match_agents = [self.agent, opp]
            winners, states = play_n_games(match_agents, self.n_games, render_mode=None)
            for i, (w, s) in enumerate(zip(winners, states)):
                all_results.append({
                    "opponent": opp.name,
                    "game_index": i,
                    "winner": w,
                    "state": s
                })
        self.results = pd.DataFrame(all_results)

    def recap(self):
        if self.results.empty:
            raise ValueError("No results. Run `self.play_matches()` first.")

        agent_name = self.agent.name
        opponents = self.results["opponent"].unique()

        summary = pd.DataFrame(index=opponents)
        summary.index.name = "opponent"

        total_games = self.n_games
        wins = self.results.groupby("opponent")["winner"].apply(lambda x: (x == agent_name).sum())
        draws = self.results.groupby("opponent")["winner"].apply(lambda x: (x == "Draw").sum())
        losses = total_games - wins - draws
        win_rate = wins / total_games
        ev = (wins - losses) / total_games

        summary["wins"] = wins
        summary["draws"] = draws
        summary["losses"] = losses
        summary["total_games"] = total_games
        summary["win_rate"] = win_rate
        summary["EV"] = ev

        return summary

    def plot_results(self, summary):
        if self.results.empty:
            raise ValueError("No results. Run `self.play_matches()` first.")

        opponents = summary.index
        wins = summary["wins"]
        draws = summary["draws"]
        losses = summary["losses"]
        total_games = summary["total_games"]
        win_rate = summary["win_rate"]
        EV = summary["EV"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Colors
        win_color = "green"
        draw_color = "yellow"
        loss_color = "red"

        # Stacked bar plot
        bars_wins = axes[0].bar(opponents, wins, label="Wins", color=win_color)
        bars_draws = axes[0].bar(opponents, draws, bottom=wins, label="Draws", color=draw_color)
        bars_losses = axes[0].bar(opponents, losses, bottom=wins + draws, label="Losses", color=loss_color)
        axes[0].set_ylabel("Games")
        axes[0].set_title(f"Performance of {self.agent.name} vs Opponents")
        axes[0].legend(loc="upper right")

        # Annotate numbers inside bars
        def annotate_bars(bars, bottom_values):
            for bar, bottom in zip(bars, bottom_values):
                height = bar.get_height()
                if height > 0:
                    axes[0].text(bar.get_x() + bar.get_width() / 2, bottom + height / 2,
                                f"{int(height)}", ha="center", va="center", color="black", fontweight="bold")
        annotate_bars(bars_wins, np.zeros(len(wins)))
        annotate_bars(bars_draws, wins)
        annotate_bars(bars_losses, wins + draws)

        # Overlay win rate line
        axes0_twin = axes[0].twinx()
        axes0_twin.plot(opponents, win_rate, color="darkblue", marker="o", linestyle="--", label="Win Rate")
        axes0_twin.set_ylabel("Win Rate")
        axes0_twin.set_ylim(0, 1)
        axes0_twin.axhline(0.5, color="red", linestyle=":", label="50% baseline")
        axes0_twin.legend(loc="upper left")

        # EV bar chart with color scale
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cmap = plt.cm.coolwarm
        ev_colors = cmap(norm(EV.values))
        bars_ev = axes[1].bar(opponents, EV, color=ev_colors)
        axes[1].set_ylabel("EV")
        axes[1].set_ylim(-1, 1)
        axes[1].set_title(f"Expected Value (EV) of {self.agent.name} vs Opponents")
        axes[1].axhline(0, color="black", linestyle="--")

        # Annotate EV values
        for bar, val in zip(bars_ev, EV.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + np.sign(val)*0.02,
                        f"{val:.2f}", ha="center", va="bottom" if val >=0 else "top", fontweight="bold", color="black")

        plt.tight_layout()
        plt.show()


    def run(self, print_recap:bool = True):
        self.play_matches()
        summary = self.recap()
        if print_recap:
            print("=== Evaluation Recap ===")
            print(summary)
        self.plot_results(summary)

    def save_results(self, recap_filepath: str|None=None, states_filepath: str|None=None):
        if self.results.empty:
            raise ValueError("No results to save. Run `self.play_matches()` first.")
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        csvname = "recap_" + self.agent.name + "_" + now_str + ".csv"
        picklename = "states_" + self.agent.name + "_" + now_str + ".pickle"
        if recap_filepath is None:
            csvpath = os.path.join(os.path.dirname(__file__), "../logs", now_str, csvname)
        else:
            csvpath = os.path.join(recap_filepath, csvname)
        if states_filepath is None:
            picklepath = os.path.join(os.path.dirname(__file__), "../logs", now_str, picklename)
        else:
            picklepath = os.path.join(states_filepath, picklename)
        os.makedirs(os.path.dirname(csvpath), exist_ok=True)
        os.makedirs(os.path.dirname(picklepath), exist_ok=True)
        summary = self.recap()
        summary.to_csv(csvpath, index=True)
        with open(picklepath, "wb") as f:
            pickle.dump(self.results, f)
