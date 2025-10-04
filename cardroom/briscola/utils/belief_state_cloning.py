import math
import torch
import numpy as np
from cardroom.briscola.game.env import BriscolaEnv
from cardroom.briscola.utils.state_encoding import encode_state

def gumbel_topk_without_replacement(logits: torch.Tensor, k: int, mask: torch.Tensor | None = None):
    """
    logits: (40,) torch float tensor (unnormalized log-probs for each card)
    k: number of items to sample without replacement
    mask: optional (40,) bool tensor; False = invalid, True = valid
    Returns: indices of length k (torch.LongTensor)
    """
    assert logits.dim() == 1 and logits.numel() == 40
    if mask is not None:
        # set invalid logits to -inf
        masked_logits = logits.clone()
        masked_logits[~mask] = float('-inf')
    else:
        masked_logits = logits

    # Draw i.i.d. Gumbel noise
    g = -torch.log(-torch.log(torch.rand_like(masked_logits)))
    # Perturb logits
    scores = masked_logits + g
    # Top-k without replacement via argsort
    topk = torch.topk(scores, k=k).indices
    return topk

def build_impossibility_mask_from_obs(obs: dict) -> np.ndarray:
    """
    Returns a boolean mask of shape (40,), True = candidate can be in opponent hand.
    Uses only observable info to avoid leakage.
    """
    # Start with all True
    can_be = np.ones(40, dtype=bool)

    # Cards in my hand are not in opponent's hand
    for card in obs["hand"]:
        can_be[card.card_id - 1] = False

    # Cards already on table cannot be in opponent's hand
    for card in obs["cards_on_table"]:
        can_be[card.card_id - 1] = False

    # Cards already played cannot be in opponent's hand
    for card in obs["all_played_cards"]:
        can_be[card.card_id - 1] = False

    # If briscola_spy is revealed/known, it cannot be in opponent hand if currently in deck bottom;
    # in classic Briscola the spy is known and at bottom; remove from opponent hand candidates.
    spy = obs["briscola_spy"]
    can_be[spy.card_id - 1] = False

    return can_be

def opponent_hand_size_from_state(state: dict, current_player_id: int) -> int:
    """
    Determine the opponent hand size exactly from state for cloning; do not leak in real play.
    """
    opp_id = 1 - current_player_id
    return len(state["hands"][opp_id])

def probabilistic_clone_from_observation(env, model, device='cpu', temperature: float = 1.0) -> BriscolaEnv:
    """
    Clone env from current player's perspective, sampling the opponent's hidden hand
    using the belief model probabilities with Gumbel-Top-k without replacement.

    - env: BriscolaEnv
    - model: BeliefMLP returning logits for 40 cards
    - device: torch device
    - temperature: optional scaling of logits before sampling
    """
    # 1) Get observation and current true state (for exact K and to copy base structures)
    obs = env.get_observation()
    state = env.get_state()

    # 2) Encode observation -> model logits
    x = encode_state(obs)  # from prior snippet
    x_t = torch.from_numpy(x).float().to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(x_t)[0].cpu()  # (40,)

    # 3) Build impossibility mask from observation
    mask_np = build_impossibility_mask_from_obs(obs)  # (40,)
    mask_t = torch.from_numpy(mask_np)

    # 4) Determine K = opponent hand size now (exact, for cloning)
    K = opponent_hand_size_from_state(state, state["current_player"])

    # 5) Temperature scaling and sampling without replacement
    scaled_logits = logits / max(1e-6, temperature)
    # Avoid all -inf: if mask rules out too many, fall back to best remaining from logits
    if mask_t.sum().item() < K:
        # relax by allowing highest-prob remaining cards even if masked; practical guard
        # Here, expand mask by enabling top (K - mask_count) invalid cards with highest logits
        deficit = int(K - mask_t.sum().item())
        invalid = (~mask_t).nonzero(as_tuple=False).flatten()
        if invalid.numel() > 0:
            _, order = torch.sort(scaled_logits[invalid], descending=True)
            enable = invalid[order[:deficit]]
            mask_t[enable] = True

    chosen_idx = gumbel_topk_without_replacement(scaled_logits, k=K, mask=mask_t)  # indices 0..39

    # 6) Build a new clone env like env.clone(), but overwrite opponent hand by sampled cards,
    # reinsert removed cards to deck to keep total multiset consistent, and preserve dealing order.
    clone = env.clone()
    clone.render_mode = None  # silence rendering

    me_id = state["current_player"]
    opp_id = 1 - me_id

    # Collect all cards that are not fixed: we'll reconstruct opponent hand from sampled ids,
    # ensuring no duplication and consistency with cards_on_table, my hand, played, and deck/spy.
    def card_by_id(card_id):
        # linear search across known universe in dealer deck + players + table + spy
        # Simpler: dealer has object instances; we must pick from deck if available; otherwise,
        # from wherever that id currently resides. We'll retrieve references consistently.
        # Build map id->object from clone structures.
        return id2card[card_id]

    # Build an id->card map from all locations in clone
    id2card = {}
    # dealer deck (top at index 0, spy at index 0 in your code when reinserted)
    for c in clone.dealer.deck:
        id2card[c.card_id] = c
    # spy card (redundant if already in deck at pos 0; ensure mapping)
    id2card[clone.briscola_spy.card_id] = clone.briscola_spy
    # players’ hands and taken cards
    for p in clone.players:
        for c in p.hand:
            id2card[c.card_id] = c
        for c in p.taken_cards:
            id2card[c.card_id] = c
    # table and played history
    for c in clone.cards_on_table:
        id2card[c.card_id] = c
    for c in clone.played_cards_history:
        id2card[c.card_id] = c

    # Remove opponent's current unknown hand from clone and return them to dealer deck tail,
    # so we can assemble the sampled set from available pool.
    # Note: If deck_left > 0 in Briscola, spy should remain as deck[0]; we won't touch it.
    opp_cards_prev = list(clone.players[opp_id].hand)
    clone.players[opp_id].hand.clear()

    # Ensure removed cards go back to deck end to preserve current top order (spy at 0)
    # and not disturb near-future draws except for the opponent hand composition.
    # Remove duplicates: if a removed card is already in deck due to previous code, skip.
    existing_ids_in_deck = set(c.card_id for c in clone.dealer.deck)
    for c in opp_cards_prev:
        if c.card_id not in existing_ids_in_deck:
            clone.dealer.deck.append(c)
            existing_ids_in_deck.add(c.card_id)

    # Now materialize sampled opponent hand
    sampled_cards = []
    used_ids = set()
    # Respect observation constraints: ensure we never pick a card violating the mask at this point.
    for idx in chosen_idx.tolist():
        cid = idx + 1
        if not mask_np[idx]:
            # Should not happen due to mask handling; skip defensively
            continue
        if cid in used_ids:
            continue
        # Fetch the card object; if not in deck, it must be somewhere fixed and thus invalid.
        card_obj = id2card.get(cid, None)
        if card_obj is None:
            continue
        # Ensure the card is not in my hand / table / played; mask already prevents, but double-check.
        illegal = False
        for c in clone.players[me_id].hand:
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
        if illegal:
            continue
        sampled_cards.append(card_obj)
        used_ids.add(cid)
        # If the card was in deck, remove it from deck to avoid duplications
        # Keep spy at position 0 untouched
        for i, dc in enumerate(clone.dealer.deck):
            if dc.card_id == cid:
                # pop from deck
                clone.dealer.deck.pop(i)
                break

        if len(sampled_cards) == K:
            break

    # Fallback if we couldn't collect K due to inconsistencies: fill with highest-prob valid leftovers from deck
    if len(sampled_cards) < K:
        need = K - len(sampled_cards)
        # Build list of valid deck cards (excluding spy at 0)
        valid_deck_cards = []
        for i, dc in enumerate(clone.dealer.deck):
            if i == 0 and dc.card_id == clone.briscola_spy.card_id:
                continue
            if mask_np[dc.card_id - 1] and (dc.card_id not in used_ids):
                valid_deck_cards.append(dc)
        # Sort by logits desc
        valid_deck_cards.sort(key=lambda c: float(logits[c.card_id - 1]), reverse=True)
        sampled_cards.extend(valid_deck_cards[:need])
        # Remove those from deck
        remove_ids = set(c.card_id for c in valid_deck_cards[:need])
        clone.dealer.deck = [c for c in clone.dealer.deck if c.card_id not in remove_ids or c.card_id == clone.briscola_spy.card_id]

    # Assign opponent hand
    clone.players[opp_id].hand = sampled_cards

    # Keep everything else unchanged and stop rendering
    clone.render_mode = None
    return clone
