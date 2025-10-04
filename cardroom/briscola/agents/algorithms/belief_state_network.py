import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-norm residual MLP block
        h = self.fc1(self.norm1(x))
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(self.norm2(h))
        return x + h

class BeliefMLP(nn.Module):
    """
    Input: encoded_state(state) -> shape [B, 246] (float32)
    Output: logits for 40 cards -> shape [B, 40]
    Use with BCEWithLogitsLoss for multi-label targets.
    """
    def __init__(self, in_dim=246, width=512, depth=2, penultimate=256, dropout=0.1):
        super().__init__()
        # Stem + input projection for residual path alignment
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, width)
        self.in_act = nn.GELU()
        self.in_drop = nn.Dropout(dropout)

        # Residual trunk
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout=dropout) for _ in range(depth)])

        # Penultimate compression
        self.mid_norm = nn.LayerNorm(width)
        self.mid_fc = nn.Linear(width, penultimate)
        self.mid_act = nn.GELU()
        self.mid_drop = nn.Dropout(dropout)

        # Output head to 40 logits (use BCEWithLogitsLoss)
        self.head = nn.Linear(penultimate, 40)

    def forward(self, x):
        # x: [B, 246]
        x = self.in_proj(self.in_norm(x))
        x = self.in_act(x)
        x = self.in_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.mid_fc(self.mid_norm(x))
        x = self.mid_act(x)
        x = self.mid_drop(x)

        logits = self.head(x)  # [B, 40]
        return logits

    def predict_proba(self, x):
        # Convenience: probabilities in [0,1]
        return torch.sigmoid(self.forward(x))


# model = BeliefMLP(in_dim=246, width=512, depth=2, penultimate=256, dropout=0.1)
# criterion = nn.BCEWithLogitsLoss(reduction='none')  # will apply a per-card mask
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# # batch
# # x: [B, 246] from encode_state(...)
# # y: [B, 40] from opponent_hand_output(...).astype(float32)
# # mask: [B, 40] binary mask where 1=learn, 0=ignore impossible cards this state

# logits = model(x)
# loss_raw = criterion(logits, y)          # [B, 40]
# loss = (loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()
