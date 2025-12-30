import os
import time
import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, List as TList

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-5-nano"

INPUT_CSV = "../dataset_filtered.csv"  # must contain column "text"
OUTPUT_CSV = "dataset_filtered_corrected_all.csv"

BATCH_INPUT_TEMPLATE = "batch_input_{idx:03d}.jsonl"
BATCH_OUTPUT_TEMPLATE = "batch_output_{idx:03d}.jsonl"

POLL_INTERVAL_SEC = 30

MAX_TOKENS_PER_BATCH = 1_500_000
CHARS_PER_TOKEN = 4.0
PROMPT_OVERHEAD_TOKENS = 250


def parse_corrections(raw: str) -> List[Tuple[str, str]]:
    """Parse correction pairs from model output (JSON or line-based)."""
    raw = (raw or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("corrections"), list):
            pairs = []
            for item in data["corrections"]:
                if not isinstance(item, dict):
                    continue
                frm = str(item.get("from", "")).strip()
                to = str(item.get("to", "")).strip()
                if frm and to and frm != to:
                    pairs.append((frm, to))
            if pairs:
                return pairs
    except Exception:
        pass

    pairs = []
    for line in raw.splitlines():
        line = line.strip().strip(" ,")
        if not line:
            continue
        if "," in line:
            frm, to = line.split(",", 1)
        elif "->" in line:
            frm, to = line.split("->", 1)
        else:
            continue

        frm = frm.strip().strip('"').strip("'")
        to = to.strip().strip('"').strip("'")
        if frm and to and frm != to:
            pairs.append((frm, to))

    dedup = {}
    for frm, to in pairs:
        dedup[frm] = to
    return list(dedup.items())


def consolidate_corrections(corrections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Resolve multiple candidates for the same source string via majority vote.
    Ties are broken by last occurrence; longer source strings are applied first.
    """
    by_from = defaultdict(list)
    for frm, to in corrections:
        if frm and to and frm != to:
            by_from[frm].append(to)

    final = []
    for frm, tos in by_from.items():
        counts = Counter(tos)
        max_freq = max(counts.values())
        candidates = [t for t, c in counts.items() if c == max_freq]
        for t in reversed(tos):
            if t in candidates:
                final.append((frm, t))
                break

    final.sort(key=lambda p: len(p[0]), reverse=True)
    return final


def is_token_like(s: str) -> bool:
    """Heuristic: treat as token-like if it’s alnum + simple punctuation/spaces."""
    return bool(re.fullmatch(r"[A-Za-z0-9'\- ]+", s))


def replace_one_case_insensitive(text: str, frm: str, to: str) -> str:
    """Case-insensitive replacement; prefer token-boundary matches for token-like strings."""
    escaped = re.escape(frm)
    if is_token_like(frm):
        pattern = r"(?i)(?<!\w)" + escaped + r"(?!\w)"
    else:
        pattern = r"(?i)" + escaped

    new_text, count = re.subn(pattern, to, text)
    if count == 0 and is_token_like(frm):
        new_text = re.sub(r"(?i)" + escaped, to, text)
    return new_text


def apply_corrections_to_text(text: str, corrections: List[Tuple[str, str]]) -> str:
    """Apply ordered corrections to text."""
    fixed = text
    for frm, to in corrections:
        if frm and to and frm != to:
            fixed = replace_one_case_insensitive(fixed, frm, to)
    return fixed


def build_metadata_from_row(row: pd.Series) -> str:
    """Serialize entity metadata used to ground spelling corrections."""
    t1_name = str(row.get("team1_name", "") or "")
    t2_name = str(row.get("team2_name", "") or "")
    t1_players = str(row.get("team1_players", "") or "")
    t2_players = str(row.get("team2_players", "") or "")
    t1_champs = str(row.get("team1_champions", "") or "")
    t2_champs = str(row.get("team2_champions", "") or "")

    return (
        "Metadata for the game:\n"
        f"Team 1 name: {t1_name}\n"
        f"Team 2 name: {t2_name}\n"
        f"Team 1 players: {t1_players}\n"
        f"Team 2 players: {t2_players}\n"
        f"Team 1 champions: {t1_champs}\n"
        f"Team 2 champions: {t2_champs}\n"
    )


def build_prompt_for_row(row: pd.Series) -> str:
    """Build a per-row prompt asking for string-level correction pairs only."""
    text = str(row.get("text", "") or "")
    metadata = build_metadata_from_row(row)

    return (
        "You are given a full ASR transcript from a single League of Legends game cast.\n\n"
        "Identify transcription errors and propose string-level replacements.\n\n"
        "Rules:\n"
        "- Use only the METADATA below to ground entity spelling.\n"
        "- Only include corrections that appear in the transcript.\n"
        "- Do not rewrite the transcript; output only correction pairs.\n\n"
        "Output format:\n"
        "- One correction per line as: `incorrect string, corrected string`\n"
        "- No extra commentary.\n\n"
        "METADATA:\n"
        f"{metadata}\n"
        "TRANSCRIPT START\n"
        f"{text}\n"
        "TRANSCRIPT END\n"
    )


def approximate_tokens_for_row(text: str) -> int:
    """Rough token estimate for batch budgeting (text tokens + fixed overhead)."""
    char_count = len(text or "")
    text_tokens = int(char_count / CHARS_PER_TOKEN)
    return text_tokens + PROMPT_OVERHEAD_TOKENS


def split_dataframe_into_batches(df: pd.DataFrame) -> TList[TList[int]]:
    """Group row indices into batches under MAX_TOKENS_PER_BATCH (approximate)."""
    df = df.reset_index(drop=True)
    batches: TList[TList[int]] = []
    current_batch: TList[int] = []
    current_tokens = 0

    for idx, row in df.iterrows():
        text = str(row.get("text", "") or "")
        row_tokens = approximate_tokens_for_row(text)

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

    print(f"Split {len(df)} rows into {len(batches)} batches.")
    return batches


def create_batch_input_file(df: pd.DataFrame, row_indices: TList[int], jsonl_path: str) -> None:
    """Write one /v1/responses task per row into a batch JSONL file."""
    tasks = []
    for idx in row_indices:
        row = df.iloc[idx]
        prompt = build_prompt_for_row(row)

        body = {
            "model": MODEL,
            "instructions": (
                "You are an ASR post-processor for League of Legends commentary transcripts. "
                "Output only string-level correction pairs in the required format."
            ),
            "input": prompt,
        }

        tasks.append(
            {
                "custom_id": f"row-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in tasks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Created batch input {jsonl_path} with {len(tasks)} requests.")


def run_single_batch(jsonl_path: str, batch_idx: int) -> str:
    """Upload input JSONL, run a batch job, poll, and download the output JSONL."""
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"[Batch {batch_idx}] Uploaded input file: {batch_file.id}")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"[Batch {batch_idx}] Created batch job: {batch.id}")

    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"[Batch {batch_idx}] Status: {batch.status}")
        if batch.status in ("completed", "failed", "cancelled"):
            break
        time.sleep(POLL_INTERVAL_SEC)

    if batch.status != "completed":
        print(f"[Batch {batch_idx}] Batch failed. Status: {batch.status}")
        if getattr(batch, "error_file_id", None):
            try:
                err_bytes = client.files.content(batch.error_file_id).content
                print(err_bytes.decode("utf-8", errors="ignore"))
            except Exception as e:
                print(f"Could not download error file: {e}")
        raise RuntimeError(f"Batch {batch_idx} did not complete successfully.")

    result_bytes = client.files.content(batch.output_file_id).content
    out_path = BATCH_OUTPUT_TEMPLATE.format(idx=batch_idx)
    with open(out_path, "wb") as f:
        f.write(result_bytes)

    print(f"[Batch {batch_idx}] Downloaded results to {out_path}")
    return out_path


def load_batch_results(results_jsonl_path: str) -> Dict[int, str]:
    """Load batch output JSONL into a mapping {row_idx: raw_output_text}."""
    mapping: Dict[int, str] = {}
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            custom_id = obj.get("custom_id")
            if not custom_id:
                continue
            try:
                row_idx = int(custom_id.split("row-")[-1])
            except Exception:
                continue

            body = obj.get("response", {}).get("body", {})
            raw = body["output"][1]["content"][0]["text"]
            mapping[row_idx] = raw

    return mapping


def load_batches(start_id: int, length: int) -> Dict[int, str]:
    """Convenience loader for a contiguous range of batch_output_*.jsonl files."""
    all_raw_results: Dict[int, str] = {}
    for i in range(start_id, start_id + length):
        if i < 10:
            all_raw_results |= load_batch_results(f"batch_output_00{i}.jsonl")
        else:
            all_raw_results |= load_batch_results(f"batch_output_0{i}.jsonl")
    return all_raw_results


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column in {INPUT_CSV}")

    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    batches = split_dataframe_into_batches(df)

    all_raw_results: Dict[int, str] = {}

    """
    for batch_idx, row_indices in enumerate(batches, start=16):
        input_path = BATCH_INPUT_TEMPLATE.format(idx=batch_idx)
        create_batch_input_file(df, row_indices, input_path)

        result_path = run_single_batch(input_path, batch_idx)
        batch_results = load_batch_results(result_path)
        all_raw_results.update(batch_results)

    print(f"Collected results for {len(all_raw_results)} rows across {len(batches)} batches.")
    """

    all_raw_results = load_batches(1, 21)

    fixed_texts: List[str] = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        raw = all_raw_results.get(idx, "")
        if not raw:
            fixed_texts.append(text)
            continue

        pairs = consolidate_corrections(parse_corrections(raw))
        fixed_texts.append(apply_corrections_to_text(text, pairs))

    df["text"] = fixed_texts
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset with fixes to {OUTPUT_CSV}")
