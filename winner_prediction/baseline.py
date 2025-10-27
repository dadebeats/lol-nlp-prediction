import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any, Iterable, Union
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------- Helpers
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=4, dropout=0.2,
                 bidirectional=False, pooling="last"):
        super().__init__()
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


def _as_tuple_roster(x) -> Tuple[str, ...]:
    """
    Normalize roster field to a canonical tuple of strings.
    Accepts list/tuple, or string repr like "['A','B','C']".
    """
    if isinstance(x, (list, tuple)):
        return tuple(map(str, x))
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return tuple(map(str, v))
        except Exception:
            pass
        # fallback: split by comma if it's a flat string (last resort)
        return tuple(map(str.strip, x.split(",")))
    # unknown type -> cast to string
    return (str(x),)

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def _pov_result(row: pd.Series, pov_side: str) -> int:
    """
    Return 1 if POV team won, else 0.
    Assumes team1_result ∈ {0,1} or {False, True}.
    """
    r = int(row["team1_result"])
    return r if pov_side == "team1" else 1 - r

def _team_and_roster(row: pd.Series, pov_side: str) -> Tuple[str, Tuple[str, ...]]:
    if pov_side == "team1":
        return row["team1_name"], _as_tuple_roster(row["team1_players"])
    else:
        return row["team2_name"], _as_tuple_roster(row["team2_players"])

def _other_side(side: str) -> str:
    return "team2" if side == "team1" else "team1"

# ---------- Index builder

def _build_team_roster_index(df: pd.DataFrame) -> Dict[Tuple[str, Tuple[str, ...]], List[int]]:
    """
    Build a mapping: (team_name, roster_tuple) -> list of df indices (sorted by date asc)
    where that exact team+roster appears, regardless of side.
    """
    # Pre-normalize rosters for speed
    df = df.copy()
    df["__t1_roster"] = df["team1_players"].apply(_as_tuple_roster)
    df["__t2_roster"] = df["team2_players"].apply(_as_tuple_roster)

    # explode into two records per match (just to build the index)
    records = []
    for idx, row in df.iterrows():
        records.append((row["team1_name"], row["__t1_roster"], row["date"], idx))
        records.append((row["team2_name"], row["__t2_roster"], row["date"], idx))

    # group and sort by date
    index: Dict[Tuple[str, Tuple[str, ...]], List[int]] = {}
    tmp = pd.DataFrame(records, columns=["team", "roster", "date", "df_idx"])
    tmp.sort_values("date", inplace=True)

    for (team, roster), g in tmp.groupby(["team", "roster"], sort=False):
        index[(team, roster)] = g["df_idx"].tolist()

    return index

# ---------- History fetch

def _history_indices_for_match(
    df: pd.DataFrame,
    idx: int,
    pov_side: str,
    index_by_team_roster: Dict[Tuple[str, Tuple[str, ...]], List[int]],
    k: int
) -> List[int]:
    """
    For the given match index and POV side, return indices of the previous k matches
    for the same (team, roster), strictly before this match's date, sorted asc by date.
    """
    row = df.loc[idx]
    team, roster = _team_and_roster(row, pov_side)
    series = index_by_team_roster.get((team, roster), [])
    if not series:
        return []
    # Find this match within the series (by index and by date)
    # There may be multiple matches on same day; we use date to filter strictly earlier.
    this_date = row["date"]
    # take all indices in the series whose date < this_date
    earlier = [j for j in series if df.loc[j, "date"] < this_date]
    # last k of them, then keep ascending order
    return earlier[-k:]

# ---------- Default feature function (simple, safe placeholder)

def embedding_feature_fn(row: pd.Series, pov_side: str) -> np.ndarray:
    """
    Feature function that simply returns the embedding vector for the given row.
    """
    return np.array(row["embedding"], dtype=np.float32)


def outcome_feature_fn(row: pd.Series, pov_side: str) -> np.ndarray:
    """
    Each history element is [team1_win_flag, team2_win_flag],
    so for the POV side we take its own and opponent's last result (1=win, 0=loss).
    """
    pov_win = float(_pov_result(row, pov_side))
    return np.array([pov_win], dtype=np.float32)

# ---------- Dataset builder

def build_sequence_dataset(
    df: pd.DataFrame,
    k_history: int = 5,
    min_history: int = 1,                    # require at least this many; else skip sample
    pad_to_k: bool = True,                   # left-pad to fixed length if shorter than k
    feature_fn: Callable[[pd.Series, str], np.ndarray] = embedding_feature_fn,
    return_concat: bool = True               # return concatenated [team1_hist + team2_hist]
) -> Dict[str, Any]:
    """
    Build a dataset suitable for RNN/Transformer training.

    Returns a dict with:
      - "X_left":  np.ndarray of shape (N, k, D)  history for POV=team1 for each sample's left side
      - "X_right": np.ndarray of shape (N, k, D)  history for POV=team2 for each sample's right side
      - "X":       np.ndarray of shape (N, 2*k, D) if return_concat=True (concat along time)
      - "y":       np.ndarray of shape (N,)        target result from POV (1=win)
      - "meta":    list of per-sample dicts (match_idx, pov_side, team_name, roster, history_idx lists)
    Each input match yields **two** samples: POV=team1 and POV=team2.
    """
    assert "date" in df.columns and "team1_result" in df.columns, "Missing required columns."
    df = _ensure_datetime(df, "date").copy()
    df.sort_values("date", inplace=True)  # ensure chronological order overall

    # Precompute index map for (team, roster) → match indices sorted by date
    idx_map = _build_team_roster_index(df)

    samples_meta = []
    left_hist_vecs = []
    right_hist_vecs = []
    targets = []

    # Iterate matches in chronological order to avoid peeking forward
    for match_idx in df.index:
        row = df.loc[match_idx]

        # Build two POVs for this match
        povs = ["team1", "team2"]
        pov_histories = {}

        ok = True
        for pov in povs:
            hidx = _history_indices_for_match(df, match_idx, pov, idx_map, k_history)
            if len(hidx) < min_history:
                ok = False
                break
            pov_histories[pov] = hidx

        if not ok:
            continue  # skip this match if any POV doesn't have minimum history

        # Convert histories to feature sequences (oldest→newest)
        left_idx = pov_histories["team1"]
        right_idx = pov_histories["team2"]

        def to_seq(idxs: List[int], pov: str) -> List[np.ndarray]:
            return [feature_fn(df.loc[j], pov) for j in idxs]

        left_seq = to_seq(left_idx, "team1")
        right_seq = to_seq(right_idx, "team2")

        # Determine feature dim
        D = left_seq[0].shape[0] if left_seq else feature_fn(row, "team1").shape[0]

        # Pad sequences to fixed length k_history (left-pad with zeros)
        def pad(seq: List[np.ndarray]) -> np.ndarray:
            if pad_to_k and len(seq) < k_history:
                pad_len = k_history - len(seq)
                return np.vstack([np.zeros((pad_len, D), dtype=np.float32), np.vstack(seq) if seq else np.zeros((0, D), dtype=np.float32)])
            elif len(seq) > 0:
                return np.vstack(seq)
            else:
                return np.zeros((k_history, D), dtype=np.float32)

        left_arr = pad(left_seq)
        right_arr = pad(right_seq)

        # Target from team1 POV
        # we'll create **two** samples: one with POV team1, one with POV team2 for the final concat
        for pov in ["team1", "team2"]:
            y = _pov_result(row, pov)
            # Final entry to the RNN will be concatenated vector from both team history entries.
            # We'll keep both sides as separate arrays and also a concatenated timeline if desired.
            if pov == "team1":
                X_left = left_arr
                X_right = right_arr
            else:
                # For POV=team2, we keep the left=team2 history, right=team1 history (so left is always POV)
                X_left = right_arr
                X_right = left_arr

            left_hist_vecs.append(X_left)
            right_hist_vecs.append(X_right)
            targets.append(y)

            team_name, roster = _team_and_roster(row, pov)
            samples_meta.append({
                "match_idx": match_idx,
                "pov_side": pov,
                "team_name": team_name,
                "roster": roster,
                "history_indices_pov": pov_histories[pov],
                "history_indices_opp": pov_histories[_other_side(pov)],
                "match_date": row["date"]
            })

    if not left_hist_vecs:
        raise RuntimeError("No samples produced. Loosen min_history or check roster parsing.")

    X_left = np.stack(left_hist_vecs)   # (N, k, D)
    X_right = np.stack(right_hist_vecs) # (N, k, D)
    y = np.array(targets, dtype=np.int64)

    out = {
        "X_left": X_left,
        "X_right": X_right,
        "y": y,
        "meta": samples_meta,
        "k_history": k_history
    }
    if return_concat:
        # time-concat along axis=1 to shape (N, 2k, D)
        out["X"] = np.concatenate([X_left, X_right], axis=1)

    return out

if __name__ == "__main__":
    df = pd.read_parquet("dataset.parquet")
    data = build_sequence_dataset(
        df,
        k_history=5,
        min_history=5,
        pad_to_k=True,
        feature_fn=embedding_feature_fn,
        return_concat=True
    )
    X = data["X"]              # (N, 2*k, D)
    y = data["y"]              # (N,)
    print("Samples:", X.shape, "Targets:", y.shape)
    print("First sample meta:", data["meta"][0])

    X_np = data["X"].astype("float32")  # ensure float32
    y_np = data["y"].astype("int64")

    # Train/val split (chronological)
    N = X_np.shape[0]
    split = int(0.05 * N)
    train_idx = np.arange(split)
    val_idx = np.arange(split, N)
    #perm = np.random.RandomState(42).permutation(N)
    #split = int(0.8 * N)
    #train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = X_np[train_idx], y_np[train_idx]
    X_val, y_val = X_np[val_idx], y_np[val_idx]


    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)  # (N, T, D)
            self.y = torch.from_numpy(y)  # (N,)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return self.X[i], self.y[i]


    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(input_dim=384, hidden_dim=512, num_layers=6, dropout=0.2,
                           bidirectional=False, pooling="last").to(device)

    # Class imbalance handling (pos_weight for BCEWithLogitsLoss)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor([(neg / max(pos, 1))], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)  # (B, T=10, D=384)
            yb = yb.float().to(device, non_blocking=True)  # (B,)
            logits = model(xb)  # (B,)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = (logits.sigmoid() >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += xb.size(0)
        return total_loss / total, correct / total


    best_val_acc = 0.0
    epochs = 100
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        scheduler.step(va_acc)
        print(
            f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "lstm_best.pt")

    print("Best val acc:", best_val_acc)
