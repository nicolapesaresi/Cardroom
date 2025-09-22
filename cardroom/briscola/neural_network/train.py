import torch
import torch.nn.functional as F
import numpy as np
from cardroom.briscola.neural_network.model import CardPolicyNet, masked_action_from_logits
from cardroom.briscola.game.env import BriscolaEnv
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple helper to convert obs -> tensor (same as in agent)
def obs_to_tensor(obs: dict, device=DEVICE):
    import numpy as np
    N_CARDS = 40
    N_SUITS = 4
    hand_mask = np.zeros(N_CARDS, dtype=np.float32)
    for c in obs["hand"]:
        hand_mask[c] = 1.0
    table_mask = np.zeros(N_CARDS, dtype=np.float32)
    for c in obs["cards_on_table"]:
        table_mask[c] = 1.0
    played_mask = np.zeros(N_CARDS, dtype=np.float32)
    for c in obs["all_played_cards"]:
        played_mask[c] = 1.0
    bris = np.zeros(N_SUITS, dtype=np.float32)
    if obs["briscola_id"] is not None:
        bris[obs["briscola_id"]] = 1.0
    scalars = np.array([
        obs["deck_left"] / N_CARDS,
        obs["turn_counter"] / 100.0,
        obs["current_player_points"] / 120.0,
        obs["other_player_points"] / 120.0
    ], dtype=np.float32)
    vec = np.concatenate([hand_mask, table_mask, played_mask, bris, scalars], axis=0)
    return torch.tensor(vec, device=device).unsqueeze(0)

# PPO hyperparams (small)
LR = 3e-4
CLIP_EPS = 0.2
EPOCHS = 4
BATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95

# create net and optimizer
net = CardPolicyNet().to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)

# collect rollout function (single environment for simplicity)

def collect_rollout(env, net, rollout_steps=1024):
    obs = env.reset()
    obs_t = obs_to_tensor(obs)
    storage = []
    for _ in range(rollout_steps):
        with torch.no_grad():
            logits, val = net(obs_t)
        legal = torch.tensor(obs["legal_mask"], device=DEVICE).unsqueeze(0).float()
        action_t, probs = masked_action_from_logits(logits, legal, deterministic=False)
        action = int(action_t.item())
        next_obs, reward, done, info = env.step(action)
        storage.append({
            "obs_t": obs_t.squeeze(0).cpu(),
            "logits": logits.squeeze(0).cpu(),
            "action": action,
            "value": float(val.item()),
            "reward": float(reward),
            "done": bool(done),
            "legal_mask": legal.squeeze(0).cpu()
        })
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        obs_t = obs_to_tensor(obs)
    return storage

# GAE

def compute_gae(storage, last_value=0.0, gamma=GAMMA, lam=LAM):
    advs = np.zeros(len(storage), dtype=np.float32)
    returns = np.zeros(len(storage), dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for i in reversed(range(len(storage))):
        r = storage[i]["reward"]
        v = storage[i]["value"]
        done = storage[i]["done"]
        delta = r + gamma * next_value * (1 - done) - v
        gae = delta + gamma * lam * (1 - done) * gae
        advs[i] = gae
        next_value = v
    for i in range(len(storage)):
        returns[i] = advs[i] + storage[i]["value"]
    return advs, returns

# training loop

def train_loop(iters=200):
    for it in range(iters):
        env = BriscolaEnv(names=["a","b"], render_mode=None)
        storage = collect_rollout(env, net, rollout_steps=1024)
        advs, returns = compute_gae(storage)
        obs_ts = torch.stack([s["obs_t"] for s in storage]).to(DEVICE)
        actions = torch.tensor([s["action"] for s in storage], device=DEVICE)
        old_logits = torch.stack([s["logits"] for s in storage]).to(DEVICE)
        advs_t = torch.tensor(advs, device=DEVICE)
        returns_t = torch.tensor(returns, device=DEVICE)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        n_steps = obs_ts.shape[0]
        idxs = np.arange(n_steps)
        for _ in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n_steps, BATCH_SIZE):
                mb_idx = idxs[start:start+BATCH_SIZE]
                mb_obs = obs_ts[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logits = old_logits[mb_idx]
                mb_advs = advs_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                logits, values = net(mb_obs)
                logp = F.log_softmax(logits, dim=-1)
                logp_act = logp.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    old_logp = F.log_softmax(mb_old_logits, dim=-1)
                    old_logp_act = old_logp.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp_act - old_logp_act)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = F.mse_loss(values, mb_returns)
                entropy = -torch.mean((F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1))
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                opt.zero_grad()
                loss.backward()
                opt.step()

        if it % 5 == 0:
            torch.save(net.state_dict(), f"policy_iter_{it}.pt")
            print(f"Saved checkpoint at iter {it}")

if __name__ == '__main__':
    train_loop(iters=50)
