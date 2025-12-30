import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any, Iterable, Union
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from sklearn.decomposition import PCA
import matplotlib
from random_utils.visualization import save_curves
import optuna
from optuna.trial import Trial
import json

parser = argparse.ArgumentParser(description="Train LSTM on LoL match history sequences")
# Data and target parameters
parser.add_argument("--dataset", type=str, default="m-player+team_asr-corr_mdl-openai-oai_emb3_ck-t512-o256.parquet",
                    help="Path to dataset parquet file")
parser.add_argument("--feature_fn", type=str, default="embedding",
                    choices=["embedding", "target", "numerical", "all", "garbage"],
                    help="Which feature function to use")
parser.add_argument("--target_col", type=str, default="team1_result",
                    choices=["team1_result", "team1_kills", "gamelength", "kill_total"], )
# Run quantity parameters
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--runs", type=int, default=1, help="How many times to run full training.")
# Sequence modeling parameters
parser.add_argument("--k", type=int, default=8, help="Number of history matches per team")
parser.add_argument("--min_history", type=int, default=4, help="Minimum history required for a sample")
# LSTM parameters
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of LSTM")
parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in LSTM")
parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"], help="LSTM pooling mode")
# Preprocessing parameter
parser.add_argument(
    "--pca",
    type=int,
    default=0,
    help="If > 0, apply PCA to the embedding column to this dimensionality before building the dataset.",
)

parser.add_argument("--optuna_run", action="store_true", help="Run Optuna hyperparam search before final runs.")
parser.add_argument("--optuna_trials", type=int, default=30, help="Number of Optuna trials.")
parser.add_argument("--optuna_epochs", type=int, default=10, help="Epochs per Optuna trial (cheap search).")
parser.add_argument("--optuna_timeout", type=int, default=0, help="Timeout in seconds (0 = no timeout).")
parser.add_argument("--optuna_seed", type=int, default=42, help="Seed for Optuna sampler.")
parser.add_argument("--optuna_save", type=str, default="optuna_best.json", help="Where to save best params as JSON.")
parser.add_argument("--optuna_metric", type=str, default="acc", choices=["acc", "loss"],
                    help="Objective metric for classification (acc=max, loss=min). For regression it uses MAE=min.")


def apply_pca_to_embeddings(
        df: pd.DataFrame,
        emb_col: str = "embedding",
        n_components: int = 0,
) -> pd.DataFrame:
    """
    If n_components > 0, fit PCA on the entire df[emb_col] (assumed vector-like)
    and overwrite that column with reduced-dim embeddings.

    If n_components <= 0, returns df unchanged.
    """
    if n_components is None or n_components <= 0:
        return df

    # Stack embeddings into (N, D)
    emb_mat = np.vstack(df[emb_col].to_numpy())  # each cell is list/np.array

    pca = PCA(n_components=n_components, random_state=42)
    emb_reduced = pca.fit_transform(emb_mat).astype(np.float32)  # (N, n_components)

    # Write back: each row gets a 1D np.array of length n_components
    df = df.copy()
    df[emb_col] = list(emb_reduced)

    print(
        f"🔎 PCA applied on '{emb_col}': original_dim={emb_mat.shape[1]}, "
        f"new_dim={n_components}, explained_var={pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    return df


# ---------- Helpers
class LSTMClassifier(nn.Module):
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

def garbage_feature_fn() -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=100).astype(np.float32)


def embedding_feature_fn(row: pd.Series, team_name: str) -> np.ndarray:
    """
    Embedding + side indicator (team_is_team1)
    """
    if team_name == row["team1_name"]:
        team_is_team1 = 1.0
    elif team_name == row["team2_name"]:
        team_is_team1 = 0.0
    else:
        raise ValueError(f"Team name {team_name} not found in row.")

    emb = np.array(row["embedding"], dtype=np.float32)

    return np.concatenate([
        np.array([team_is_team1], dtype=np.float32),
        emb
    ])

def both_feature_fn(row: pd.Series, team_name: str) -> np.ndarray:
    """
    Historical relict. We don't use this function in the thesis.
    """
    if team_name == row["team1_name"]:
        pov_side = "team1"
        team_is_team1 = 1.0
    elif team_name == row["team2_name"]:
        pov_side = "team2"
        team_is_team1 = 0.0
    else:
        raise ValueError(f"Invalid team name: {team_name}")

    embedding = np.array(row["embedding"], dtype=np.float32)
    pov_win = float(_pov_result(row, pov_side))

    return np.concatenate([
        np.array([team_is_team1, pov_win], dtype=np.float32),
        embedding
    ])

def target_feature_fn(row: pd.Series, team_name: str, target_col: str) -> np.ndarray:
    """
    Baseline for arbitrary target column.
    Returns [team_is_team1, pov_target_value].

    pov_target_value is:
        - row[target_col]      if POV = team1
        - row[target_col for team2] if target is symmetric
        - row[target_col]      if target is a team1-based metric (like kill_diff, kill_total)
    """
    # side indicator
    if team_name == row["team1_name"]:
        pov_val = float(row[target_col])
    elif team_name == row["team2_name"]:
        # --- RULES FOR HOW TO MAP TARGETS ---
        if target_col == "team1_result":
            pov_val = 1.0 - float(row[target_col])
        elif target_col == "team1_kills":
            pov_val = float(row["team2_kills"])
        elif target_col == "kill_diff":
            pov_val = -float(row["kill_diff"])
        elif target_col == "kill_total":
            pov_val = float(row["kill_total"])
        elif target_col == "gamelength":
            pov_val = float(row["gamelength"])
        else:
            raise ValueError(f"Unsupported target_col={target_col}")
    else:
        raise ValueError(f"Invalid team name in row.")

    return np.array([pov_val], dtype=np.float32)

def numerical_feature_fn(row: pd.Series, team_name: str) -> np.ndarray:
    """
    Use numerical match stats (no embeddings), POV-aware.
    Keeps features that have (near) no NaNs in your column list; skips early-game @10 and elders.

    Output is a 1D float32 vector.
    """
    if team_name == row["team1_name"]:
        side = "team1"
        opp = "team2"
        team_is_team1 = 1.0
    elif team_name == row["team2_name"]:
        side = "team2"
        opp = "team1"
        team_is_team1 = 0.0
    else:
        raise ValueError(f"Invalid team name: {team_name}")

    # Helper to safely cast to float
    def f(x, default=0.0) -> float:
        try:
            if pd.isna(x):
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    # Core per-team stats (low NaN counts in your list)
    kills = f(row[f"{side}_kills"])

    dragons = f(row[f"{side}_dragons"])
    barons = f(row[f"{side}_barons"])
    towers = f(row[f"{side}_towers"])

    dpm = f(row[f"{side}_damage_per_min"])
    wpm = f(row[f"{side}_wards_per_min"])
    gpm = f(row[f"{side}_gold_per_min"])

    # Opponent same stats (to let model learn relative strength without us hardcoding diffs only)
    kills_opp = f(row[f"{opp}_kills"])
    dragons_opp = f(row[f"{opp}_dragons"])
    barons_opp = f(row[f"{opp}_barons"])
    towers_opp = f(row[f"{opp}_towers"])
    dpm_opp = f(row[f"{opp}_damage_per_min"])
    wpm_opp = f(row[f"{opp}_wards_per_min"])
    gpm_opp = f(row[f"{opp}_gold_per_min"])

    # Global match-level stats you already computed (not POV-dependent)
    kill_total = f(row.get("kill_total", np.nan))
    kill_diff_abs = f(row.get("kill_diff", np.nan))

    # Side as a numeric indicator (blue=0/red=1, or whatever you prefer)
    # Use team1_side/team2_side fields if present; otherwise fallback to team_is_team1
    side_str = row.get(f"{side}_side", None)
    if isinstance(side_str, str):
        # common encodings
        side_str_l = side_str.lower()
        if side_str_l in ("blue", "b", "0"):
            side_id = 0.0
        elif side_str_l in ("red", "r", "1"):
            side_id = 1.0
        else:
            side_id = team_is_team1  # fallback
    else:
        side_id = team_is_team1

    # A small set of engineered relative features (safe + informative)
    kill_diff_signed = kills - kills_opp
    baron_diff = barons - barons_opp
    dragon_diff = dragons - dragons_opp
    tower_diff = towers - towers_opp
    gpm_diff = gpm - gpm_opp
    dpm_diff = dpm - dpm_opp

    feats = np.array([
        # identifiers / context
        team_is_team1,
        side_id,

        # raw team stats
        kills,
        dragons, barons, towers,
        dpm, wpm, gpm,

        # raw opponent stats
        kills_opp,
        dragons_opp, barons_opp, towers_opp,
        dpm_opp, wpm_opp, gpm_opp,

        # engineered relative stats
        kill_diff_signed,
        baron_diff,
        dragon_diff,
        tower_diff,
        gpm_diff,
        dpm_diff,

        # global match context
        kill_total,
        kill_diff_abs,
    ], dtype=np.float32)

    return feats


def all_feature_fn(row: pd.Series, team_name: str) -> np.ndarray:
    """
    Concatenate numerical_feature_fn + embedding_feature_fn.
    """
    num = numerical_feature_fn(row, team_name)
    emb = embedding_feature_fn(row, team_name)
    return np.concatenate([num, emb]).astype(np.float32)


# ---------- Dataset builder

def build_sequence_dataset(
    df: pd.DataFrame,
    k_history: int = 8,
    min_history: int = 4,  # require at least this many; else skip sample
    pad_to_k: bool = True,  # left-pad to fixed length if shorter than k
    feature_fn: Callable[[pd.Series, str], np.ndarray] = embedding_feature_fn,
    target_column: str = "team1_result",
) -> Dict[str, Any]:
    """
    Build a dataset suitable for RNN/Transformer training.

    Returns a dict with:
      - "X_left":  np.ndarray of shape (N, k, D)  history for POV=team1 for each sample's left side
      - "X_right": np.ndarray of shape (N, k, D)  history for POV=team2 for each sample's right side
      - "X":       np.ndarray of shape (N, 2*k, D)
      - "y":       np.ndarray of shape (N,)        target result from POV (1=win)
      - "meta":    list of per-sample dicts (match_idx, pov_side, team_name, roster, history_idx lists)
    Each input match yields two samples: POV=team1 and POV=team2, which we concatenate to a **single** vector.
    """
    assert "date" in df.columns and "team1_result" in df.columns, "Missing required columns."
    df = _ensure_datetime(df, "date").copy()
    df.sort_values("date", inplace=True)  # ensure chronological order overall

    # Precompute index map for (team, roster) → match indices sorted by date
    idx_map = _build_team_roster_index(df)

    seqs = []  # NEW: store (k, D_total) sequences
    targets = []
    samples_meta = []

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

        def to_seq(idxs: List[int], team_name: str) -> List[np.ndarray]:
            return [feature_fn(df.loc[j], team_name) for j in idxs]

        left_seq = to_seq(left_idx, row["team1_name"])
        right_seq = to_seq(right_idx, row["team2_name"])
        # Determine feature dim
        D = left_seq[0].shape[0] if left_seq else feature_fn(row, "team1").shape[0]

        # Pad sequences to fixed length k_history (left-pad with zeros)
        def pad(seq: List[np.ndarray]) -> np.ndarray:
            if pad_to_k and len(seq) < k_history:
                pad_len = k_history - len(seq)
                return np.vstack([np.zeros((pad_len, D), dtype=np.float32),
                                  np.vstack(seq) if seq else np.zeros((0, D), dtype=np.float32)])
            elif len(seq) > 0:
                return np.vstack(seq)
            else:
                return np.zeros((k_history, D), dtype=np.float32)

        left_arr = pad(left_seq)
        right_arr = pad(right_seq)

        y = int(row[target_column])

        # Per-timestep feature concat: (k, D_left + D_right)
        pair_seq = np.concatenate([left_arr, right_arr], axis=1)  # <-- feature axis

        seqs.append(pair_seq)
        targets.append(y)

        samples_meta.append({
            "match_idx": match_idx,
            "pov_side": "team1",
            "team_name": row["team1_name"],
            "roster": _as_tuple_roster(row["team1_players"]),
            "history_indices_pov": left_idx,
            "history_indices_opp": right_idx,
            "match_date": row["date"]
        })

    if not seqs:
        raise RuntimeError("No samples produced. Loosen min_history or check roster parsing.")

    X = np.stack(seqs)  # (N, k, D_total)
    y = np.array(targets, dtype=np.int64)

    out = {"X": X, "y": y, "meta": samples_meta, "k_history": k_history}
    return out

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    is_classification: bool
) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    metric_sum = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        if is_classification:
            preds = (logits.sigmoid() >= 0.5).long()
            metric_sum += (preds == yb.long()).sum().item()  # correct
        else:
            metric_sum += (logits - yb).abs().sum().item()  # MAE sum

    avg_loss = total_loss / max(n, 1)
    avg_metric = metric_sum / max(n, 1)  # accuracy or MAE
    return avg_loss, avg_metric

class SeqDataset(Dataset):
    def __init__(self, X_arr: np.ndarray, y_arr: np.ndarray):
        self.X = torch.from_numpy(X_arr)
        self.y = torch.from_numpy(y_arr)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]

# --- helper: global seeding for reproducibility ---
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --- helper: build model/criterion/optimizer/scheduler from params ---
def build_training_objects(
    *,
    input_dim: int,
    device: torch.device,
    is_classification: bool,
    y_train_np: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    pooling: str,
    lr: float,
    weight_decay: float,
):
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        pooling=pooling,
    ).to(device)

    if is_classification:
        pos = int((y_train_np == 1).sum())
        neg = int((y_train_np == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # we pass transformed metric (acc or -MAE / -loss)
        factor=0.9,
        patience=3,
        min_lr=1e-5,
    )
    return model, criterion, optimizer, scheduler


# --- helper: train loop (shared by optuna + final runs) ---
def train_eval(
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    is_classification: bool,
    epochs: int,
    save_path: Optional[str] = None,
    trial: Optional[Trial] = None,
    objective_mode: str = "acc",   # "acc" or "loss" for classification; regression uses MAE
) -> Tuple[float, Dict[str, List[float]]]:
    best = -np.inf
    curves = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metric = run_epoch(
            model=model, loader=train_loader, criterion=criterion, optimizer=optimizer,
            device=device, is_classification=is_classification
        )
        va_loss, va_metric = run_epoch(
            model=model, loader=val_loader, criterion=criterion, optimizer=None,
            device=device, is_classification=is_classification
        )

        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(va_loss)
        curves["train_metric"].append(tr_metric)
        curves["val_metric"].append(va_metric)

        # transform to "maximize" always
        if is_classification:
            if objective_mode == "loss":
                score = -va_loss
            else:
                score = va_metric  # accuracy
        else:
            score = -va_metric     # -MAE

        scheduler.step(score)

        if score > best:
            best = score
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        # optuna pruning support
        if trial is not None:
            trial.report(best, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best, curves


if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.optuna_seed)

    # --- feature function selection (late-bind args.target_col safely)
    if args.feature_fn == "embedding":
        feat_fn = embedding_feature_fn
    elif args.feature_fn == "garbage":
        feat_fn = lambda row, team: garbage_feature_fn()
    elif args.feature_fn == "both":
        feat_fn = both_feature_fn
    elif args.feature_fn == "target":
        feat_fn = lambda row, team: target_feature_fn(row, team, args.target_col)
    elif args.feature_fn == "numerical":
        feat_fn = lambda row, team: numerical_feature_fn(row, team)
    elif args.feature_fn == "all":
        feat_fn = lambda row, team: all_feature_fn(row, team)
    else:
        raise ValueError(f"Unknown feature_fn={args.feature_fn}")

    print(
        f"Using feature_fn={args.feature_fn}, k={args.k}, "
        f"hidden_dim={args.hidden_dim}, layers={args.num_layers}, pooling={args.pooling}, "
        f"bidirectional={args.bidirectional}"
    )
    print(f"Predicting target_col={args.target_col}")

    # --- load + enrich
    df = pd.read_parquet(args.dataset)
    df = df.sort_values("date")
    print("Original raw data size: ", len(df))
    df["kill_diff"] = (df["team1_kills"] - df["team2_kills"]).abs()
    df["kill_total"] = df["team1_kills"] + df["team2_kills"]

    # --- optional PCA
    df = apply_pca_to_embeddings(df, emb_col="embedding", n_components=args.pca)

    # --- dataset build (chronological by builder)
    data = build_sequence_dataset(
        df,
        k_history=args.k,
        min_history=args.min_history,
        pad_to_k=True,
        feature_fn=feat_fn,
        return_concat=True,
        target_column=args.target_col,
    )

    X = data["X"].astype(np.float32)
    y_raw = data["y"]
    print("Samples:", X.shape, "Targets:", y_raw.shape)
    print("First sample meta:", data["meta"][0])

    is_classification = (args.target_col == "team1_result")
    y = y_raw.astype(np.float32)

    # --- chronological split
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # --- data loaders (rebuilt per trial so batch_size can vary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            SeqDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            SeqDataset(X_val, y_val),
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader

    # -------------------------
    # Optuna search (optional)
    # -------------------------
    if args.optuna_run:
        # Note: we maximize `score` defined in train_eval (acc or -loss / -MAE)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)

        def objective(trial: Trial) -> float:
            # define search ranges
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 192, 256, 384, 512])
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.35)
            bidirectional = trial.suggest_categorical("bidirectional", [False])
            pooling = trial.suggest_categorical("pooling", ["last", "mean"])

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 96, 128])
            train_loader, val_loader = make_loaders(batch_size)

            model, criterion, optimizer, scheduler = build_training_objects(
                input_dim=X.shape[2],
                device=device,
                is_classification=is_classification,
                y_train_np=y_train,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                pooling=pooling,
                lr=lr,
                weight_decay=weight_decay,
            )

            # temp checkpoint per trial (optional)
            ckpt_path = None  # or f"optuna_trial_{trial.number}.pt"

            score, _ = train_eval(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                is_classification=is_classification,
                epochs=args.optuna_epochs,
                save_path=ckpt_path,
                trial=trial,
                objective_mode=args.optuna_metric,
            )
            return score

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            objective,
            n_trials=args.optuna_trials,
            timeout=None if args.optuna_timeout <= 0 else args.optuna_timeout,
            show_progress_bar=True,
        )

        print("Optuna best score:", study.best_value)
        print("Optuna best params:", study.best_params)

        # overwrite args.* with best params (so the final args.runs use them)
        best = study.best_params
        args.hidden_dim = int(best.get("hidden_dim", args.hidden_dim))
        args.num_layers = int(best.get("num_layers", args.num_layers))
        args.dropout = float(best.get("dropout", args.dropout))
        args.bidirectional = bool(best.get("bidirectional", args.bidirectional))
        args.pooling = str(best.get("pooling", args.pooling))
        args.lr = float(best.get("lr", args.lr))
        args.weight_decay = float(best.get("weight_decay", args.weight_decay))
        args.batch_size = int(best.get("batch_size", args.batch_size))

        # save best params to json for reproducibility
        payload = {
            "best_score": float(study.best_value),
            "best_params": study.best_params,
            "optuna_trials": args.optuna_trials,
            "optuna_epochs": args.optuna_epochs,
            "objective": args.optuna_metric,
        }
        with open(args.optuna_save, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved Optuna best params to {args.optuna_save}")

        # rebuild loaders with tuned batch size
        train_loader, val_loader = make_loaders(args.batch_size)

    else:
        train_loader, val_loader = make_loaders(args.batch_size)

    # -------------------------
    # Final runs (args.runs)
    # -------------------------
    best_metrics = []
    for run in range(args.runs):
        model, criterion, optimizer, scheduler = build_training_objects(
            input_dim=X.shape[2],
            device=device,
            is_classification=is_classification,
            y_train_np=y_train,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            pooling=args.pooling,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        ckpt = f"lstm_best_{args.feature_fn}_{args.target_col}_run{run+1}.pt"
        best_val_metric, curves = train_eval(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            is_classification=is_classification,
            epochs=args.epochs,
            save_path=ckpt,
            trial=None,
            objective_mode=args.optuna_metric,
        )

        # save plots
        epochs_arr = np.arange(1, args.epochs + 1)
        save_curves(
            epochs=epochs_arr,
            train_losses=curves["train_loss"],
            val_losses=curves["val_loss"],
            train_metrics=curves["train_metric"],
            val_metrics=curves["val_metric"],
            is_classification=is_classification,
            feature_fn=args.feature_fn,
            target_col=args.target_col,
        )

        print(f"Run {run+1}/{args.runs} complete. Best val score: {best_val_metric:.3f}")
        best_metrics.append(best_val_metric)

    vals = np.array(best_metrics, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    print(f"mean = {mean:.6f}, std = {std:.6f}")
