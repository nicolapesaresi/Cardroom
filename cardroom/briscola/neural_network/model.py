import torch 
import torch.nn as nn
import torch.nn.functional as F

class CardPolicyNet(nn.Module):
    def __init__(self, n_cards: int = 40, n_suits: int = 4, hidden=256):
        super().__init__()
        self.n_cards = n_cards
        # input: hand_mask, table_mask, played_mask, briscola_onehot, 4 scalars
        self.input_dim = 3 * n_cards + n_suits + 4
        self.fc1 = nn.Linear(self.input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, n_cards)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs_tensor):
        x = F.relu(self.fc1(obs_tensor))
        x = F.relu(self.fc2(x))
        return self.logits(x), self.value(x).squeeze(-1)

# helper: masked action selection

def masked_action_from_logits(logits: torch.Tensor, legal_mask: torch.Tensor, deterministic=False):
    very_neg = -1e9
    masked_logits = logits.clone()
    masked_logits[legal_mask == 0] = very_neg
    probs = F.softmax(masked_logits, dim=-1)
    if deterministic:
        action = torch.argmax(probs, dim=-1)
    else:
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return action, probs