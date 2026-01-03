"""
Build Pass 2 training/evaluation samples from Pass 1 event extractions.

This script converts a match-level dataset that contains:
- match metadata (date, teams, rosters, outcome),
- an events column produced by Pass 1 (list of {label,text} dicts),

into a JSONL dataset where each record contains:
- a TARGET match,
- each team's k-match history (oldest -> newest),
- the influential event lists for each historical match (POV-aligned).

History construction mirrors the baseline roster-aware indexing logic used in
`predictions/run_experiment.py`: history is retrieved for the same
(team_name, roster) and restricted to matches strictly earlier than the target.
"""

import argparse
import ast
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _as_tuple_roster(x: Any) -> Tuple[str, ...]:
    """Normalize roster representations into a canonical tuple of strings."""
    if isinstance(x, (list, tuple)):
        return tuple(map(str, x))
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return tuple(map(str, v))
        except Exception:
            pass
        return tuple(map(str.strip, x.split(",")))
    return (str(x),)


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure `df[col]` is datetime64 (UTC), coercing invalid entries to NaT."""
    if col not in df.columns:
        raise ValueError(f"Missing required datetime column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _pov_result(row: pd.Series, pov_side: str) -> int:
    """Return 1 if the POV side won the match, else 0 (based on team1_result)."""
    r = int(row["team1_result"])
    return r if pov_side == "team1" else 1 - r


def _team_and_roster(row: pd.Series, pov_side: str) -> Tuple[str, Tuple[str, ...]]:
    """Return (team_name, roster_tuple) for the requested POV side."""
    if pov_side == "team1":
        return row["team1_name"], _as_tuple_roster(row["team1_players"])
    return row["team2_name"], _as_tuple_roster(row["team2_players"])


def _build_team_roster_index(df: pd.DataFrame) -> Dict[Tuple[str, Tuple[str, ...]], List[int]]:
    """
    Build an index mapping (team_name, roster_tuple) -> df indices, sorted by date asc.
    This supports roster-consistent history retrieval.
    """
    df = df.copy()
    df["__t1_roster"] = df["team1_players"].apply(_as_tuple_roster)
    df["__t2_roster"] = df["team2_players"].apply(_as_tuple_roster)

    records = []
    for idx, row in df.iterrows():
        records.append((row["team1_name"], row["__t1_roster"], row["date"], idx))
        records.append((row["team2_name"], row["__t2_roster"], row["date"], idx))

    tmp = pd.DataFrame(records, columns=["team", "roster", "date", "df_idx"])
    tmp.sort_values("date", inplace=True)

    index: Dict[Tuple[str, Tuple[str, ...]], List[int]] = {}
    for (team, roster), g in tmp.groupby(["team", "roster"], sort=False):
        index[(team, roster)] = g["df_idx"].tolist()
    return index


def _history_indices_for_match(
    df: pd.DataFrame,
    match_idx: int,
    pov_side: str,
    index_by_team_roster: Dict[Tuple[str, Tuple[str, ...]], List[int]],
    k: int,
) -> List[int]:
    """
    Retrieve the previous k matches for the same (team, roster), strictly before the target match date.
    Returned indices are in chronological order (oldest -> newest).
    """
    row = df.loc[match_idx]
    team, roster = _team_and_roster(row, pov_side)
    series = index_by_team_roster.get((team, roster), [])
    if not series:
        return []

    this_date = row["date"]
    earlier = [j for j in series if df.loc[j, "date"] < this_date]
    return earlier[-k:]


def _is_nonempty_events(v: Any) -> bool:
    """Return True iff v parses to a non-empty list of events."""
    if v is None:
        return False
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "[]", "null", "None"):
            return False
        try:
            parsed = ast.literal_eval(s)
            return isinstance(parsed, list) and len(parsed) > 0
        except Exception:
            try:
                parsed = json.loads(s)
                return isinstance(parsed, list) and len(parsed) > 0
            except Exception:
                return False
    return False


def _parse_events(v: Any) -> List[Dict[str, Any]]:
    """
    Parse events into a list of dicts.
    Accepts: already-list, python literal string, or JSON string.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "[]", "null", "None"):
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def _find_events_col(df: pd.DataFrame) -> str:
    """
    Find a plausible Pass 1 events column.
    Prefers known names; otherwise falls back to any column containing 'events'.
    """
    preferred = [
        "pass1_events",
        "events",
        "match_events",
        "dataset_pass1_events",
        "extracted_events",
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if "events" in c.lower():
            return c
    raise ValueError("Could not find an events column (expected pass1_events / events / *events*).")


def build_sequence_jsonl(
    df: pd.DataFrame,
    *,
    events_col: str,
    k_history: int,
    min_history: int,
    include_rendered_prompt: bool,
    max_events_per_match: int,
) -> List[Dict[str, Any]]:
    """
    Construct Pass 2 samples for each eligible target match.

    Each output record includes:
    - `target`: match metadata + label (team1_result),
    - `history`: team1 and team2 histories, POV-aligned,
    - optional `rendered_prompt` for downstream LLM pipelines.

    Filtering:
    - matches with empty event lists are excluded,
    - targets where either team lacks `min_history` roster-consistent matches are excluded.
    """
    df = _ensure_datetime(df, "date").copy()
    df.sort_values("date", inplace=True)

    df = df.loc[df[events_col].apply(_is_nonempty_events)].copy()
    df["__events"] = df[events_col].apply(_parse_events)

    idx_map = _build_team_roster_index(df)
    out: List[Dict[str, Any]] = []

    for match_idx in df.index:
        row = df.loc[match_idx]

        left_idx = _history_indices_for_match(df, match_idx, "team1", idx_map, k_history)
        right_idx = _history_indices_for_match(df, match_idx, "team2", idx_map, k_history)

        if len(left_idx) < min_history or len(right_idx) < min_history:
            continue

        def hist_item(j: int, target_team: str) -> Dict[str, Any]:
            r = df.loc[j]

            if r["team1_name"] == target_team:
                opponent = r["team2_name"]
                pov_win = int(r["team1_result"])
            else:
                opponent = r["team1_name"]
                pov_win = 1 - int(r["team1_result"])

            ev = r["__events"]
            if max_events_per_match > 0:
                ev = ev[:max_events_per_match]

            return {
                "match_idx": int(j) if isinstance(j, (int, np.integer)) else str(j),
                "date": r["date"].isoformat() if pd.notna(r["date"]) else None,
                "team": target_team,
                "opponent": opponent,
                "pov_win": int(pov_win),
                "events": ev,
            }

        team1_hist = [hist_item(j, row["team1_name"]) for j in left_idx]
        team2_hist = [hist_item(j, row["team2_name"]) for j in right_idx]

        target = {
            "match_idx": int(match_idx) if isinstance(match_idx, (int, np.integer)) else str(match_idx),
            "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
            "team1_name": row["team1_name"],
            "team2_name": row["team2_name"],
            "team1_players": _as_tuple_roster(row["team1_players"]),
            "team2_players": _as_tuple_roster(row["team2_players"]),
            "team1_result": int(row["team1_result"]),
            "events_present": True,
        }

        rec: Dict[str, Any] = {
            "target": target,
            "k_history": k_history,
            "min_history": min_history,
            "history": {"team1": team1_hist, "team2": team2_hist},
        }

        if include_rendered_prompt:
            rec["rendered_prompt"] = render_prompt(rec)

        out.append(rec)

    return out


def render_prompt(rec: Dict[str, Any]) -> str:
    """
    Render a compact, deterministic prompt from a Pass 2 record.
    This is optional and can be regenerated later if prompt formatting changes.
    """
    t = rec["target"]
    lines = [f"Target match: {t['team1_name']} vs {t['team2_name']} ({t['date']})", ""]

    for side in ("team1", "team2"):
        team_name = t[f"{side}_name"]
        lines.append(f"{team_name} recent matches (oldest -> newest):")
        for h in rec["history"][side]:
            lines.append(f"- vs {h['opponent']} on {h['date']} | pov_win={h['pov_win']}")
            for ev in h["events"]:
                lines.append(f"  * [{ev.get('label','')}] {ev.get('text','')}")
        lines.append("")

    lines.append("Predict: will team1 win? Answer with 0 or 1.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="dataset_with_pass1_events_2025.csv")
    ap.add_argument("--out", type=str, default="sequence_pass2.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min_history", type=int, default=4)
    ap.add_argument("--events_col", type=str, default="", help="If empty, auto-detect *events* column.")
    ap.add_argument("--include_rendered_prompt", action="store_true")
    ap.add_argument(
        "--max_events_per_match",
        type=int,
        default=0,
        help="If >0, keep only first N events per historical match (token budgeting).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    events_col = args.events_col.strip() if args.events_col.strip() else _find_events_col(df)

    seq = build_sequence_jsonl(
        df,
        events_col=events_col,
        k_history=args.k,
        min_history=args.min_history,
        include_rendered_prompt=args.include_rendered_prompt,
        max_events_per_match=args.max_events_per_match,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in seq:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(seq)} samples to {args.out}")
    print(f"Used events_col='{events_col}'")
    if len(seq) > 0:
        print("Example keys:", list(seq[0].keys()))
