"""
PASS 1: Convert a full League of Legends commentary transcript into an ordered
list of influential match events.

Each event is a continuous substring of the original transcript and is assigned
exactly one label from:
[teamfight, objective_kill, team_performance, player_performance, match_winner].

The resulting event sequences are later used as structured inputs for downstream
prediction models. This script performs batch inference via the OpenAI Responses
API and stores parsed event lists alongside the original dataset.
"""

import os
import time
import json
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-5"

INPUT_PATH = "../m-player+team_asr-corr_mdl-openai-oai_emb3_ck-t512-o256.parquet"
OUTPUT_CSV = "dataset_with_pass1_events_2025.csv"

BATCH_INPUT_TEMPLATE = "batch_input_pass1_{idx:03d}.jsonl"
BATCH_OUTPUT_TEMPLATE = "batch_output_pass1_{idx:03d}.jsonl"

POLL_INTERVAL_SEC = 30
MAX_TOKENS_PER_BATCH = 1_500_000
CHARS_PER_TOKEN = 4.0
PROMPT_OVERHEAD_TOKENS = 350

ALLOWED_LABELS = [
    "teamfight",
    "objective_kill",
    "team_performance",
    "player_performance",
    "match_winner",
]


def build_metadata_from_row(row: pd.Series) -> str:
    """
    Construct a compact metadata block used to ground entity references
    (teams, players, champions) during event extraction.
    """
    return (
        "Metadata (may be partial):\n"
        f"- league: {row.get('league', '')}\n"
        f"- patch: {row.get('patch', '')}\n"
        f"- date: {row.get('date', '')}\n"
        f"- team1: {row.get('team1_name', '')}\n"
        f"- team2: {row.get('team2_name', '')}\n"
        f"- team1_players: {row.get('team1_players', '')}\n"
        f"- team2_players: {row.get('team2_players', '')}\n"
        f"- team1_champions: {row.get('team1_champions', '')}\n"
        f"- team2_champions: {row.get('team2_champions', '')}\n"
    )


def build_pass1_prompt_for_row(row: pd.Series) -> str:
    """
    Build the Pass 1 prompt requesting an ordered, labeled list of influential
    events from a full commentary transcript.
    """
    text = str(row.get("text", "") or "")
    metadata = build_metadata_from_row(row)

    return (
        "You are given a full commentary transcript from a single League of Legends match.\n\n"
        "Task: Extract an ORDERED list of the most influential events that explain how the game unfolded.\n"
        "Each item must be a continuous substring of the original text, in chronological order.\n\n"
        "You must assign EXACTLY one label per item from the following set:\n"
        f"{ALLOWED_LABELS}\n\n"
        "Guidelines:\n"
        "- Prefer macro-informative events (teamfights, objectives, tempo swings).\n"
        "- Avoid filler, repetition, or micro play-by-play.\n"
        "- Do not invent events or discuss future matches.\n"
        "- Include exactly one 'match_winner' item near the end if the winner is clear.\n"
        "- Aim for 10–20 events to allow reconstruction of the match flow.\n\n"
        "Output format (STRICT JSON ONLY):\n"
        "{\n"
        '  "events": [\n'
        '    {"label": "...", "text": "..."}\n'
        "  ]\n"
        "}\n\n"
        "METADATA:\n"
        f"{metadata}\n"
        "TRANSCRIPT START\n"
        f"{text}\n"
        "TRANSCRIPT END\n"
    )


def approximate_tokens_for_row(text: str) -> int:
    """Heuristic token estimate used for batch budgeting."""
    return int(len(text or "") / CHARS_PER_TOKEN) + PROMPT_OVERHEAD_TOKENS


def split_dataframe_into_batches(df: pd.DataFrame) -> List[List[int]]:
    """
    Split dataset row indices into batches such that the approximate total
    token count per batch does not exceed MAX_TOKENS_PER_BATCH.
    """
    df = df.reset_index(drop=True)
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_tokens = 0

    for idx, row in df.iterrows():
        row_tokens = approximate_tokens_for_row(str(row.get("text", "") or ""))

        if row_tokens > MAX_TOKENS_PER_BATCH:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([idx])
            continue

        if current_tokens + row_tokens > MAX_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(idx)
        current_tokens += row_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def create_batch_input_file(df: pd.DataFrame, row_indices: List[int], jsonl_path: str) -> None:
    """
    Create a batch JSONL file containing one Responses API request per dataset row.
    """
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx in row_indices:
            body = {
                "model": MODEL,
                "instructions": (
                    "Extract an ordered list of influential League of Legends match events. "
                    "Output must be STRICT JSON only."
                ),
                "input": build_pass1_prompt_for_row(df.iloc[idx]),
            }
            task = {
                "custom_id": f"row-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def run_single_batch(jsonl_path: str, batch_idx: int) -> str:
    """
    Upload a batch input file, execute the batch job, poll until completion,
    and download the resulting output JSONL.
    """
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    while True:
        batch = client.batches.retrieve(batch.id)
        if batch.status in ("completed", "failed", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_idx} failed with status {batch.status}")

    out_path = BATCH_OUTPUT_TEMPLATE.format(idx=batch_idx)
    result_bytes = client.files.content(batch.output_file_id).content
    with open(out_path, "wb") as f:
        f.write(result_bytes)

    return out_path


def _extract_output_text_from_batch_line(obj: Dict[str, Any]) -> str:
    """
    Extract raw text output from a single batch result line, handling
    multiple possible Responses API output layouts.
    """
    body = obj.get("response", {}).get("body", {}) or {}
    output = body.get("output")

    if isinstance(output, list):
        for item in output:
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(c, dict) and "text" in c:
                    return c["text"]

    try:
        return body["output"][1]["content"][0]["text"]
    except Exception:
        return ""


def parse_pass1_events(raw: str) -> List[Dict[str, str]]:
    """
    Parse a strict JSON event list, enforcing allowed labels and preserving order.
    """
    try:
        data = json.loads((raw or "").strip())
    except Exception:
        return []

    events = data.get("events", [])
    if not isinstance(events, list):
        return []

    out = []
    for ev in events:
        if (
            isinstance(ev, dict)
            and ev.get("label") in ALLOWED_LABELS
            and isinstance(ev.get("text"), str)
            and ev["text"].strip()
        ):
            out.append({"label": ev["label"], "text": ev["text"].strip()})

    return out


def load_batch_results(results_jsonl_path: str) -> Dict[int, str]:
    """
    Load a batch output JSONL file into a mapping from row index to raw output text.
    """
    mapping: Dict[int, str] = {}
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            try:
                idx = int(cid.split("row-")[-1])
            except Exception:
                continue
            mapping[idx] = _extract_output_text_from_batch_line(obj)
    return mapping


if __name__ == "__main__":

    df = pd.read_parquet(INPUT_PATH)
    # For our evaluation we only include data past GPT 4 training cutoff
    df = df[df.date > "2024-06"].reset_index(drop=True).tail(660)

    batches = split_dataframe_into_batches(df)
    all_raw_results: Dict[int, str] = {}

    for batch_idx, row_indices in enumerate(batches):
        input_path = BATCH_INPUT_TEMPLATE.format(idx=batch_idx)
        create_batch_input_file(df, row_indices, input_path)
        result_path = run_single_batch(input_path, batch_idx)
        all_raw_results.update(load_batch_results(result_path))

    pass1_events_col: List[str] = []
    for idx in range(len(df)):
        events = parse_pass1_events(all_raw_results.get(idx, ""))
        pass1_events_col.append(json.dumps(events, ensure_ascii=False))

    df["pass1_events"] = pass1_events_col
    df.to_csv(OUTPUT_CSV, index=False)
