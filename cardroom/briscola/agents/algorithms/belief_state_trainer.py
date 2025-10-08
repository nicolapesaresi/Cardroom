import os
import time
import json
import math
import copy
import random
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import wandb  # pip install wandb

from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.agents.donatello import DonatelloAgent
from cardroom.briscola.agents.random import RandomAgent
from cardroom.briscola.utils.state_encoding import encode_state, opponent_hand_output
from cardroom.briscola.agents.algorithms.belief_state_network import BeliefMLP


@dataclass
class TrainerConfig:
    project: str = "briscola-belief"
    run_name: Optional[str] = None
    out_dir: str = "./runs/briscola_belief"
    device: str = "cuda"
    seed: int = 42

    # self-play collection
    episodes_per_iter: int = 500
    max_steps_per_ep: int = 50  # safety
    determinization_temp: float = 1.0  # if using belief in ISMCTS, not used here

    # training
    batch_size: int = 1024
    iters: int = 200
    epochs_per_iter: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    pos_weight: Optional[List[float]] = None  # length 40 or None
    val_split: float = 0.1

    # logging/checkpoint
    log_wandb: bool = True
    save_every_iter: int = 1
    early_stop_patience: int = 20


class SelfPlayBeliefTrainer:
    def __init__(self, model: nn.Module, env_factory, agent_factory_pair, cfg: TrainerConfig):
        """
        model: BeliefMLP-like network returning logits (B,40)
        env_factory: callable -> BriscolaEnv
        agent_factory_pair: callable -> (agent0, agent1) for self-play
        cfg: TrainerConfig
        """
        self.model = model
        self.env_factory = env_factory
        self.agent_factory_pair = agent_factory_pair
        self.cfg = cfg

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.device = device
        self.model.to(self.device)

        # optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # pos_weight tensor for BCE
        self.pos_weight = None
        if cfg.pos_weight is not None:
            self.pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=self.device)

        # state
        self.global_step = 0
        self.iter_idx = 0
        self.best_val = float("inf")
        self.best_state = None

        # dirs
        self.out_dir = pathlib.Path(cfg.out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.plot_dir = self.out_dir / "plots"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # rng
        # random.seed(cfg.seed)
        # np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # wandb
        self.wandb_run = None
        if cfg.log_wandb:
            self.wandb_run = wandb.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg), save_code=True, reinit=True)
            wandb.watch(self.model, log="all", log_freq=100)

    def save_checkpoint(self, tag: str):
        path = self.ckpt_dir / f"{tag}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_idx": self.iter_idx,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "cfg": asdict(self.cfg),
        }
        torch.save(payload, path)
        if self.wandb_run is not None:
            wandb.save(str(path))  # mark as artifact in W&B
        return str(path)

    def load_checkpoint(self, path: str):
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iter_idx = payload.get("iter_idx", 0)
        self.global_step = payload.get("global_step", 0)
        self.best_val = payload.get("best_val", float("inf"))
        return payload

    def _split_train_val(self, X, Y, M, val_ratio):
        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        val_n = int(n * val_ratio)
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]
        return (X[tr_idx], Y[tr_idx], M[tr_idx]), (X[val_idx], Y[val_idx], M[val_idx])

    @torch.no_grad()
    def _evaluate(self, Xv, Yv, Mv):
        self.model.eval()
        bs = self.cfg.batch_size
        total_loss = 0.0
        total_count = 0
        for i in range(0, len(Xv), bs):
            xb = torch.from_numpy(Xv[i:i+bs]).float().to(self.device)
            yb = torch.from_numpy(Yv[i:i+bs]).float().to(self.device)
            mb = torch.from_numpy(Mv[i:i+bs]).float().to(self.device)
            logits = self.model(xb)
            loss_elem = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none", pos_weight=self.pos_weight)
            loss = (loss_elem * mb).sum() / mb.sum().clamp_min(1.0)
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
        return total_loss / max(1, total_count)

    def _log_plots(self, metrics: Dict[str, float], iter_idx: int):
        # simple Matplotlib loss curve saved to disk and optionally logged to W&B
        # Accumulate metrics in a JSONL file to plot history
        hist_path = self.out_dir / "history.jsonl"
        with open(hist_path, "a") as f:
            f.write(json.dumps({"iter": iter_idx, **metrics}) + "\n")

        # load history to plot
        xs, train_losses, val_losses = [], [], []
        with open(hist_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                xs.append(rec["iter"])
                train_losses.append(rec["train_loss"])
                val_losses.append(rec["val_loss"])

        plt.figure(figsize=(6,4))
        plt.plot(xs, train_losses, label="train")
        plt.plot(xs, val_losses, label="val")
        plt.xlabel("iteration")
        plt.ylabel("BCE (masked)")
        plt.title("Belief training losses")
        plt.legend()
        fig_path = self.plot_dir / f"loss_iter_{iter_idx:05d}.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        if self.wandb_run is not None:
            wandb.log({"train/loss": metrics["train_loss"], "val/loss": metrics["val_loss"], "charts/loss_curve": wandb.Image(str(fig_path))}, step=self.global_step)

    def _build_mask_from_obs(self, obs: dict) -> np.ndarray:
        mask = np.ones(40, dtype=np.float32)
        for c in obs["hand"]:
            mask[c.card_id - 1] = 0.0
        for c in obs.get("cards_on_table", []):
            mask[c.card_id - 1] = 0.0
        for c in obs.get("all_played_cards", []):
            mask[c.card_id - 1] = 0.0
        if obs["turn_counter"] < 18: # mask spy only if it's not yet drawn
            spy = obs["briscola_spy"]
            mask[spy.card_id - 1] = 0.0
        return mask

    def _play_and_collect(self, episodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y, M = [], [], []
        print("Playing games...")
        for ep in tqdm(range(episodes)):
            env = self.env_factory()
            agent0, agent1 = self.agent_factory_pair()
            agents = [agent0, agent1]

            step_guard = 0
            while not env.done and step_guard < self.cfg.max_steps_per_ep:
                obs = env.get_observation()
                x = encode_state(obs)
                # label uses full state (self-play supervision)
                state = env.get_state()
                opp_id = 1 - obs["current_player"]
                y = opponent_hand_output(state, opp_id)  # 40-d one-hot/multi-hot
                m = self._build_mask_from_obs(obs)

                X.append(x.astype(np.float32))
                Y.append(y.astype(np.float32))
                M.append(m.astype(np.float32))

                # select action from agent based on obs
                cur = env.current_player_id
                agent = agents[cur]
                legal = env.get_legal_actions()
                action = None
                try:
                    if isinstance(agent, DonatelloAgent):
                        action = agent.select_action(obs, env)
                    else:
                        action = agent.select_action(obs)
                except Exception:
                    pass
                if action is None or int(action) not in legal:
                    action = random.choice(legal)
                env.step(int(action))
                step_guard += 1

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        M = np.array(M, dtype=np.float32)
        return X, Y, M

    def _train_epoch(self, Xt, Yt, Mt):
        self.model.train()
        bs = self.cfg.batch_size
        n = len(Xt)
        # shuffle in-place
        idx = np.arange(n)
        np.random.shuffle(idx)
        Xt, Yt, Mt = Xt[idx], Yt[idx], Mt[idx]

        total = 0.0
        count = 0
        for i in range(0, n, bs):
            xb = torch.from_numpy(Xt[i:i+bs]).float().to(self.device)
            yb = torch.from_numpy(Yt[i:i+bs]).float().to(self.device)
            mb = torch.from_numpy(Mt[i:i+bs]).float().to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(xb)
            loss_elem = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none", pos_weight=self.pos_weight)
            loss = (loss_elem * mb).sum() / mb.sum().clamp_min(1.0)
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total += loss.item() * xb.size(0)
            count += xb.size(0)
            self.global_step += 1

        return total / max(1, count)

    def train(self):
        for it in range(self.iter_idx, self.cfg.iters):
            self.iter_idx = it

            # 1) collect
            X, Y, M = self._play_and_collect(self.cfg.episodes_per_iter)

            # 2) split
            (Xt, Yt, Mt), (Xv, Yv, Mv) = self._split_train_val(X, Y, M, self.cfg.val_split)

            # 3) epochs
            train_loss = None
            for ep in range(self.cfg.epochs_per_iter):
                train_loss = self._train_epoch(Xt, Yt, Mt)

            # 4) evaluate
            val_loss = self._evaluate(Xv, Yv, Mv)

            metrics = {"train_loss": float(train_loss if train_loss is not None else 0.0),
                       "val_loss": float(val_loss)}
            self._log_plots(metrics, it)

            # 5) checkpoint last
            last_path = self.save_checkpoint("last")
            # 6) if best, checkpoint best
            if val_loss < self.best_val:
                self.best_val = val_loss
                best_path = self.save_checkpoint("best")
                # keep a copy of best state dict in memory for fast reload if needed
                self.best_state = copy.deepcopy(self.model.state_dict())
                if self.wandb_run is not None:
                    wandb.summary["best_val_loss"] = self.best_val
                    wandb.log({"val/best_loss": self.best_val}, step=self.global_step)

            # optional: early stopping based on patience
            if (it - np.argmin([json.loads(l)["val_loss"] for l in open(self.out_dir / "history.jsonl")])) >= self.cfg.early_stop_patience:
                # break early if patience exceeded
                break

        # end train
        if self.wandb_run is not None:
            self.wandb_run.finish()


# Example factories for environment and agents
def make_env():
    return BriscolaEnv(render_mode=None)

def make_agents():
    a0, a1 = DonatelloAgent(simulations=200), DonatelloAgent(simulations=200)
    a0.belief_model = model; a0.use_belief = True; a0.det_temp = 1.0
    a1.belief_model = model; a1.use_belief = True; a1.det_temp = 1.0

    return a0, a1


if __name__ == "__main__":

    model = BeliefMLP()  # from earlier
    cfg = TrainerConfig(run_name="v1", out_dir="./runs/briscola_belief_v1", episodes_per_iter=300, iters=500)
    trainer = SelfPlayBeliefTrainer(model, env_factory=make_env, agent_factory_pair=make_agents, cfg=cfg)
    trainer.train()
