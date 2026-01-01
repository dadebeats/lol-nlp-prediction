import numpy as np
import pandas as pd

def _pov_result(row: pd.Series, pov_side: str) -> int:
    """
    Return 1 if POV team won, else 0.
    Assumes team1_result ∈ {0,1} or {False, True}.
    """
    r = int(row["team1_result"])
    return r if pov_side == "team1" else 1 - r

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


def target_feature_fn(row: pd.Series, team_name: str, target_col: str) -> np.ndarray:
    """
    Baseline for arbitrary target column.
    Handles the fact that our team could be the team_2 (flips; or takes the value of the opposing team)
    Returns [pov_target_value].
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
