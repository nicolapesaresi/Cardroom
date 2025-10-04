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

class AgentEvaluator:
    """Evaluates agent play by simulating games against a range of opponents and computeing stats."""
    def __init__(self, agent: Agent, n_games: int, opponents: list[Agent] | None = None):
        """Initializes evaluator.
        Args:
            agent: agent to evaluate.
            n_games: number of games to play against each agent.
            opponents: list of opponents.
        """
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

    def play_matches(self):
        """Plays the matches against the opponents, capturing original/flipped pairing."""
        all_results = []
        pair_global = 0
        for opp in self.opponents:
            print(f"Opponent: {opp.name}")
            match_agents = [self.agent, opp]
            opp_results = []  # this opponent's games
            # Each call returns two games: original and flipped
            with tqdm(total=self.n_games, desc="Playing games...") as pbar:
                prev_n = 0
                for j in range(self.n_games // 2):
                    winners, states = play_flipped_games(match_agents, render_mode=None)

                    pair_id = f"{pair_global}"
                    pair_global += 1

                    for i, (w, s) in enumerate(zip(winners, states)):
                        game_res = {
                            "opponent": opp.name,
                            "game_index": j * 2 + i,
                            "winner": w,
                            "final_state": s,
                            "pair_id": pair_id
                        }
                        opp_results.append(game_res)
                    curr_n = len(opp_results)
                    pbar.update(curr_n - prev_n)
                    prev_n = curr_n
                    a0_name, a1_name = match_agents[0].name, match_agents[1].name
                    a0_wins = sum(1 for r in opp_results if r["winner"] == a0_name)
                    a1_wins = sum(1 for r in opp_results if r["winner"] == a1_name)
                    draws   = sum(1 for r in opp_results if r["winner"] == "Draw")
                    desc_str = f"{a0_name} wins: {a0_wins} | {a1_name} wins: {a1_wins} | Draws: {draws}"
                    pbar.set_description(f"Playing games... ({desc_str})", refresh=False)
            all_results.extend(opp_results)

        self.results = pd.DataFrame(all_results)

    def recap(self) -> pd.DataFrame:
        """Computes final statistics, including significance tests."""
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

        # --- significance test ---
        p_values = []
        significant = []
        for opp in opponents:
            w = wins.loc[opp]
            n = total_games - draws.loc[opp]  # exclude draws for the test
            if n > 0:
                res = binomtest(w, n, p=0.5, alternative="two-sided")
                p_values.append(res.pvalue)
                significant.append(res.pvalue < 0.05)
            else:
                p_values.append(np.nan)
                significant.append(False)

        summary["wins"] = wins
        summary["draws"] = draws
        summary["losses"] = losses
        summary["total_games"] = total_games
        summary["win_rate"] = win_rate
        summary["EV"] = ev
        summary["p_value"] = p_values
        summary["significant"] = significant

        return summary

    def plot_results(self, summary):
        """Plots the results and returns the plot."""
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
        cmap = plt.cm.RdYlGn
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
        return fig

    def evaluate_endgames(self, n_endgames: int=500):
        """Evaluates agent's endgame ability comparing with minimax optimal strategy."""
        evaluator = EndgameEvaluator(self.agent)
        results, fig = evaluator.evaluate_agent(n_endgames)
        return results, fig


    def analyze_flipped(self) -> pd.DataFrame:
        """Analyze original vs flipped paired results per opponent (McNemar on win/loss)."""
        if self.results.empty:
            raise ValueError("No results. Run `self.play_matches()` first.")

        agent = self.agent.name
        out_rows = []

        # Each pair_id under an opponent should have exactly 2 rows
        for opponent, opp_rows in self.results.groupby("opponent"):
            # build per-pair labels
            labels = []
            for pair_id, g in opp_rows.groupby("pair_id"):
                winners = g["winner"].tolist()
                if len(winners) != 2:
                    # skip malformed pairs; or raise if strict
                    continue

                a_cnt = winners.count(agent)
                o_cnt = winners.count(opponent)
                d_cnt = winners.count("Draw")

                if a_cnt == 2:
                    label = "won_both"
                elif o_cnt == 2:
                    label = "lost_both"
                elif d_cnt == 2:
                    label = "draw_draw"  # counted in tot_pairs but not in the four agent-tilt buckets
                elif d_cnt == 1 and a_cnt == 1:
                    label = "won_drew"
                elif d_cnt == 1 and o_cnt == 1:
                    label = "lost_drew"
                elif a_cnt == 1 and o_cnt == 1:
                    label = "one_each"
                else:
                    label = "unknown"

                labels.append(label)

            # aggregate counts
            tot_pairs = len(labels)
            counts = pd.Series(labels).value_counts()
            row = {
                "opponent": opponent,
                "tot_pairs": tot_pairs,
                "won_both": int(counts.get("won_both", 0)),
                "lost_both": int(counts.get("lost_both", 0)),
                "won_drew": int(counts.get("won_drew", 0)),
                "lost_drew": int(counts.get("lost_drew", 0)),
                "one_each": int(counts.get("one_each", 0)),
                "draw_draw": int(counts.get("draw_draw", 0)),
                "unknown": int(counts.get("unknown", 0)),
            }
            out_rows.append(row)

        summary = pd.DataFrame(out_rows).set_index("opponent").sort_index()
        return summary

    def plot_flipped_analysis(self, summary: pd.DataFrame):
        """One figure with two subplots:
        - Left: Concordant vs Discordant counts per opponent
            concordant = one_each + draw_draw
            discordant = tot_pairs - concordant
        - Right: Discordant breakdown per opponent
            positive = won_both + won_drew
            negative = lost_both + lost_drew
        Returns the Matplotlib Figure.
        """
        required = {"tot_pairs","won_both","lost_both","won_drew","lost_drew","one_each","draw_draw"}
        missing = required - set(summary.columns)
        if missing:
            raise ValueError(f"Summary missing columns: {missing}")

        df = summary.copy()
        df["concordant"] = df["one_each"] + df["draw_draw"]
        df["discordant"] = df["tot_pairs"] - df["concordant"]
        wb = df["won_both"]
        wd = df["won_drew"]
        ld = df["lost_drew"]
        lb = df["lost_both"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        # Left subplot: Concordant vs Discordant
        ax1 = axes[0]
        x = np.arange(len(df.index))
        ax1.bar(df.index, df["concordant"], label="Concordant", color="#6baed6")
        ax1.bar(df.index, df["discordant"], bottom=df["concordant"], label="Discordant", color="#fd8d3c")
        ax1.set_title("Concordant vs Discordant by Opponent")
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Opponent")
        ax1.legend()
        for xi, c, d in zip(x, df["concordant"].tolist(), df["discordant"].tolist()):
            if c > 0:
                ax1.text(xi, c/2, str(int(c)), ha="center", va="center", color="white", fontsize=9)
            if d > 0:
                ax1.text(xi, c + d/2, str(int(d)), ha="center", va="center", color="white", fontsize=9)
        ax1.set_axisbelow(True)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        # Right subplot: Discordant breakdown
        ax2 = axes[1]
        # Start with zeros as the baseline
        bottom0 = np.zeros(len(df.index))

        # Positive stacks (greens)
        bars_wb = ax2.bar(df.index, wb, bottom=bottom0, label="Won both", color="green")
        bars_wd = ax2.bar(df.index, wd, bottom=wb, label="Won + Draw", color="#90ee90")

        # Compute current bottom after positives
        bottom_pos = wb + wd

        # Negative stacks (reds), stacked on top of positives to show full discordant total
        bars_ld = ax2.bar(df.index, ld, bottom=bottom_pos, label="Lost + Draw", color="coral")
        bars_lb = ax2.bar(df.index, lb, bottom=bottom_pos + ld, label="Lost both", color="red")

        ax2.set_title("Discordant Pairs Breakdown")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Opponent")
        ax2.legend()

        # Annotations centered in each segment
        x = np.arange(len(df.index))
        for xi, a, b, c, d in zip(x, wb.tolist(), wd.tolist(), ld.tolist(), lb.tolist()):
            if a > 0:
                ax2.text(xi, a/2, str(int(a)), ha="center", va="center", color="white", fontsize=9)
            if b > 0:
                ax2.text(xi, a + b/2, str(int(b)), ha="center", va="center", color="white", fontsize=9)
            if c > 0:
                ax2.text(xi, a + b + c/2, str(int(c)), ha="center", va="center", color="white", fontsize=9)
            if d > 0:
                ax2.text(xi, a + b + c + d/2, str(int(d)), ha="center", va="center", color="white", fontsize=9)

        ax2.set_axisbelow(True)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        # Rotate labels if many opponents
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)

        return fig

    def run(self, print_recap:bool = True):
        """Runs the evaluation pipeline: plays the games, calculates and plots results, saves results."""
        self.play_matches()
        summary = self.recap()
        if print_recap:
            print("=== Evaluation Recap ===")
            print(summary)

        self.flipped_summary = self.analyze_flipped()
        if print_recap:
            print("=== Flipped-side Analysis ===")
            print(self.flipped_summary)
            
        fig = self.plot_results(summary)
        fig_flip = self.plot_flipped_analysis(self.flipped_summary)

        print("Evaluating endgames...")
        self.endgames, endgamefig = self.evaluate_endgames()

        self.save_results(matchesplot = fig, endgameplot= endgamefig, flipplot = fig_flip)

    def save_results(self, folder: str|None=None, matchesplot=None, endgameplot=None, flipplot=None):
        """Saves results."""
        if self.results.empty:
            raise ValueError("No results to save. Run `self.play_matches()` first.")
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        csvname = "recap_" + self.agent.name + "_" + now_str + ".csv"
        picklename = "states_" + self.agent.name + "_" + now_str + ".pickle"
        if folder is None:
            folder = os.path.join(os.path.dirname(__file__), "../logs", now_str)
        os.makedirs(folder, exist_ok=True)
        csvpath = os.path.join(folder, "matches.csv")
        resultspath = os.path.join(folder, "games.csv")
        picklepath = os.path.join(folder, "final_states.pickle")
        matchesplotpath = os.path.join(folder, "matches.png")
        endgamepickle = os.path.join(folder, "endgames.pickle")
        endgameplotpath = os.path.join(folder, "endgames.png")
        flippedpath = os.path.join(folder, "flipped_games.csv")
        flippedplotpath = os.path.join(folder, "flipped_games.png")

        self.results.to_csv(resultspath)
        summary = self.recap()
        summary.to_csv(csvpath, index=True)
        with open(picklepath, "wb") as f:
            pickle.dump(self.results, f)
        with open(endgamepickle, "wb") as f:
            pickle.dump(self.endgames, f)

        if matchesplot is not None:
            matchesplot.savefig(matchesplotpath)
        if endgameplot is not None:
            endgameplot.savefig(endgameplotpath)
        self.flipped_summary.to_csv(flippedpath, index=True)
        if flipplot is not None:
            flipplot.savefig(flippedplotpath)