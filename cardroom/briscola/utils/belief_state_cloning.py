import math
import torch
import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.utils.state_encoding import encode_state

import torch
import numpy as np

# Assumes BeliefMLP and encode_state(obs) are available from prior steps.
# The following utilities are adapted for use inside ISMCTS.

def gumbel_topk_without_replacement(logits: torch.Tensor, k: int, mask: torch.Tensor | None = None):
    assert logits.dim() == 1 and logits.numel() == 40
    if mask is not None:
        logits = logits.clone()
        logits[~mask] = float('-inf')
    # Draw Gumbel noise
    g = -torch.log(-torch.log(torch.rand_like(logits)))
    scores = logits + g
    return torch.topk(scores, k=k).indices  # (k,)

def build_impossibility_mask_from_obs(obs: dict) -> np.ndarray:
    can_be = np.ones(40, dtype=bool)
    for c in obs["hand"]:
        can_be[c.card_id - 1] = False
    for c in obs.get("cards_on_table", []):
        can_be[c.card_id - 1] = False
    for c in obs.get("all_played_cards", []):
        can_be[c.card_id - 1] = False
    spy = obs["briscola_spy"]
    can_be[spy.card_id - 1] = False
    return can_be

def opponent_hand_size_from_state(state: dict, current_player_id: int) -> int:
    opp_id = 1 - current_player_id
    return len(state["hands"][opp_id])

def determinize_with_belief(env, model, device='cpu', temperature: float = 1.0):
    """
    Create a determinized clone using the belief model to sample the opponent's current hand.
    """
    obs = env.get_observation()
    state = env.get_state()

    # 1) Model logits from encoded observation
    x = encode_state(obs)
    x_t = torch.from_numpy(x).float().to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(x_t)[0].cpu()  # (40,)

    # 2) Mask impossibles and set hand size K
    mask_np = build_impossibility_mask_from_obs(obs)
    mask_t = torch.from_numpy(mask_np)
    K = opponent_hand_size_from_state(state, state["current_player"])

    # 3) Temperature scaling
    scaled = logits / max(1e-6, temperature)

    # 4) If mask has fewer than K, relax by enabling best-scoring invalids
    if mask_t.sum().item() < K:
        deficit = int(K - mask_t.sum().item())
        invalid = (~mask_t).nonzero(as_tuple=False).flatten()
        if invalid.numel() > 0:
            _, order = torch.sort(scaled[invalid], descending=True)
            enable = invalid[order[:deficit]]
            mask_t[enable] = True

    chosen = gumbel_topk_without_replacement(scaled, k=K, mask=mask_t).tolist()
    chosen_ids = {i + 1 for i in chosen}  # card ids are 1..40

    # 5) Build clone and rewrite opponent hand consistently
    clone = env.clone()
    clone.render_mode = None
    me = state["current_player"]
    opp = 1 - me

    # Map id -> card object across clone structures
    id2card = {}
    for c in clone.dealer.deck:
        id2card[c.card_id] = c
    id2card[clone.briscola_spy.card_id] = clone.briscola_spy
    for p in clone.players:
        for c in p.hand:
            id2card[c.card_id] = c
        for c in p.taken_cards:
            id2card[c.card_id] = c
    for c in clone.cards_on_table:
        id2card[c.card_id] = c
    for c in clone.played_cards_history:
        id2card[c.card_id] = c

    # Remove opponent's current hand to deck tail if not already in deck
    prev = list(clone.players[opp].hand)
    clone.players[opp].hand.clear()
    present = set(c.card_id for c in clone.dealer.deck)
    for c in prev:
        if c.card_id not in present:
            clone.dealer.deck.append(c)
            present.add(c.card_id)

    # Assemble sampled hand from available pool, removing from deck as needed
    sampled = []
    used = set()
    for cid in chosen_ids:
        # double-check legality
        if not mask_np[cid - 1]:
            continue
        card_obj = id2card.get(cid, None)
        if card_obj is None:
            continue
        illegal = False
        for c in clone.players[me].hand:
            if c.card_id == cid:
                illegal = True; break
        if not illegal:
            for c in clone.cards_on_table:
                if c.card_id == cid:
                    illegal = True; break
        if not illegal:
            for c in clone.played_cards_history:
                if c.card_id == cid:
                    illegal = True; break
        if not illegal and clone.briscola_spy.card_id == cid:
            illegal = True
        if illegal or cid in used:
            continue

        sampled.append(card_obj)
        used.add(cid)
        # remove from deck if present (except spy at 0)
        for i, dc in enumerate(clone.dealer.deck):
            if dc.card_id == cid:
                clone.dealer.deck.pop(i)
                break

        if len(sampled) == K:
            break

    # Fallback: fill from valid deck cards sorted by belief
    if len(sampled) < K:
        need = K - len(sampled)
        candidates = []
        for i, dc in enumerate(clone.dealer.deck):
            if i == 0 and dc.card_id == clone.briscola_spy.card_id:
                continue
            if mask_np[dc.card_id - 1] and (dc.card_id not in used):
                candidates.append(dc)
        candidates.sort(key=lambda c: float(logits[c.card_id - 1]), reverse=True)
        take = candidates[:need]
        sampled.extend(take)
        remove_ids = set(c.card_id for c in take)
        clone.dealer.deck = [c for c in clone.dealer.deck if c.card_id not in remove_ids or c.card_id == clone.briscola_spy.card_id]

    clone.players[opp].hand = sampled
    return clone

