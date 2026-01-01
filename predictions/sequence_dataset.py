from typing import Callable, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import ast

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

def build_sequence_dataset(
    df: pd.DataFrame,
    feature_fn: Callable[[pd.Series, str], np.ndarray],
    k_history: int = 8,
    min_history: int = 4,  # require at least this many; else skip sample
    pad_to_k: bool = True,  # left-pad to fixed length if shorter than k
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
    #idx_map = _build_team_index(df)

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
            #hidx = _history_indices_for_match_relaxed(df, match_idx, pov, idx_map, k_history, max_changes=2)

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

def _team_and_roster(row: pd.Series, pov_side: str) -> Tuple[str, Tuple[str, ...]]:
    if pov_side == "team1":
        return row["team1_name"], _as_tuple_roster(row["team1_players"])
    else:
        return row["team2_name"], _as_tuple_roster(row["team2_players"])