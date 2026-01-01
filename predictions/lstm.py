import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    Class which can take a
    """
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=4, dropout=0.2,
                 bidirectional=False, pooling="last"):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # expects (B, T, D)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.pooling = pooling  # "last" or "mean"
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):  # x: (B, T, D)
        x = self.in_norm(x)   # (B, T, D)
        out, (h_n, c_n) = self.lstm(x)
        if self.pooling == "last":
            # last time step hidden (already shaped (B, T, H*dir))
            if self.lstm.bidirectional:
                # concatenate last forward and last backward hidden
                # h_n: (num_layers*dir, B, H)
                last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
            else:
                last = h_n[-1]  # (B, H)
            feats = last
        else:
            # mean pool across time steps
            feats = out.mean(dim=1)  # (B, H*dir)
        logits = self.head(feats).squeeze(-1)  # (B,)
        return logits