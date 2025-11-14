import os
import re
import time
import json
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# -------------------------------
# Config
# -------------------------------
MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o-mini",
    "o3"

    # add more here if you want to try multiple models in one run
    # "o4-mini",
]
WORDS_PER_CHUNK = 250
OVERLAP_WORDS = 10
REQUEST_DELAY_SEC = 0.2  # light pacing for rate-limit friendliness

INPUT_PATH = "asr/asr_error_correction_devset.jsonl"
OUTPUT_PATH = "asr/asr_error_correction_devset_corrected.jsonl"

# -------------------------------
# Utils: chunking
# -------------------------------
def chunk_text_by_words(text: str, words_per_chunk: int = 250, overlap_words: int = 25) -> List[str]:
    words = text.split()
    n = len(words)
    if n == 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + words_per_chunk, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        # advance with overlap
        start = max(end - overlap_words, start + 1)
    return chunks

# -------------------------------
# Utils: parsing corrections
# Accepts:
#   - lines: "incorrect string, corrected string"
#   - JSON: {"corrections":[{"from":"...", "to":"..."}]}
# Ignores empties and malformed lines.
# -------------------------------
def parse_corrections(raw: str) -> List[Tuple[str, str]]:
    raw = (raw or "").strip()
    if not raw:
        return []

    # Try JSON first (strict or lenient)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "corrections" in data and isinstance(data["corrections"], list):
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

    # Fallback: line-based "from, to"
    pairs = []
    for line in raw.splitlines():
        line = line.strip().strip(" ,")
        if not line:
            continue
        # Split only on the first comma to allow commas in the 'to' part; still brittle but works well in practice
        if "," in line:
            frm, to = line.split(",", 1)
            frm = frm.strip().strip('"').strip("'")
            to = to.strip().strip('"').strip("'")
            if frm and to and frm != to:
                pairs.append((frm, to))
        # Optional: support arrow style "from -> to"
        elif "->" in line:
            frm, to = line.split("->", 1)
            frm = frm.strip().strip('"').strip("'")
            to = to.strip().strip('"').strip("'")
            if frm and to and frm != to:
                pairs.append((frm, to))

    # Deduplicate while preserving the last occurrence for the same 'from'
    dedup = {}
    for frm, to in pairs:
        dedup[frm] = to
    return list(dedup.items())

# -------------------------------
# Utils: consolidating corrections across chunks
# Strategy:
#   - if multiple 'to' values for the same 'from', keep the most frequent; break ties by last seen.
#   - drop self-maps or trivially empty.
# -------------------------------
def consolidate_corrections(corrections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    by_from = defaultdict(list)
    for frm, to in corrections:
        if frm and to and frm != to:
            by_from[frm].append(to)

    final = []
    for frm, tos in by_from.items():
        counts = Counter(tos)
        # pick the most common 'to'; if tie, choose the last occurrence in original order
        most_common_freq = max(counts.values())
        candidates = [t for t, c in counts.items() if c == most_common_freq]
        # last-seen preference
        for t in reversed(tos):
            if t in candidates:
                final.append((frm, t))
                break

    # Sort by descending length of 'from' to avoid partial-overwrite issues during replacement
    final.sort(key=lambda p: len(p[0]), reverse=True)
    return final

# -------------------------------
# Replacement helpers
# - Prefer whole-token replacements when the 'from' looks token-like.
# - Case-insensitive matching; preserve provided 'to' as-is.
# -------------------------------
def is_token_like(s: str) -> bool:
    # If s is composed of words, spaces, apostrophes, hyphens, digits — treat as token-like
    return bool(re.fullmatch(r"[A-Za-z0-9'\- ]+", s))

def replace_one_case_insensitive(text: str, frm: str, to: str) -> str:
    # If token-like, try whole-word boundary match
    escaped = re.escape(frm)
    if is_token_like(frm):
        # boundaries around token-like phrase
        pattern = r"(?i)(?<!\w)" + escaped + r"(?!\w)"
    else:
        # raw substring, case-insensitive
        pattern = r"(?i)" + escaped

    new_text, count = re.subn(pattern, to, text)
    # If we got zero with boundaries but the string is token-like, fall back to raw case-insensitive substring
    if count == 0 and is_token_like(frm):
        pattern2 = r"(?i)" + escaped
        new_text = re.sub(pattern2, to, text)
    return new_text

def apply_corrections_to_text(text: str, corrections: List[Tuple[str, str]]) -> str:
    fixed = text
    for frm, to in corrections:
        if not frm or not to or frm == to:
            continue
        fixed = replace_one_case_insensitive(fixed, frm, to)
    return fixed

# -------------------------------
# LLM call for a single chunk
# -------------------------------
def request_corrections_for_chunk(model: str, metadata: str, chunk: str) -> List[Tuple[str, str]]:
    # Prompting style: minimal and machine-parsable
    prompt = (
        "Check the provided transcript CHUNK (not the full game) for ASR errors. "
        "Use only the metadata to ground entity corrections. "
        "Return ONLY the proposed corrections, no explanations. "
        "Format: one correction per line as 'incorrect string, corrected string'. "
        "Be exhaustive but only include corrections that actually appear in THIS CHUNK.\n\n"
        f"{metadata}\n\n"
        f"CHUNK START\n{chunk}\nCHUNK END"
    )

    response = client.responses.create(
        model=model,
        instructions=(
            "You are a precise ASR post-processor for League of Legends commentary transcripts. "
            "Output strictly the corrections in the minimal required format. "
            "Do not include any extra text besides the corrections. "
            "You will be penalized for missing any errors."
        ),
        input=prompt
    )
    raw = response.output_text
    return parse_corrections(raw)

# -------------------------------
# Row processing
# -------------------------------
def build_metadata_from_row(row: pd.Series) -> str:
    # Safely coerce to strings to avoid NaNs
    t1_name = str(row.get("team1_name", ""))
    t2_name = str(row.get("team2_name", ""))
    t1_players = str(row.get("team1_players", ""))
    t2_players = str(row.get("team2_players", ""))
    t1_champs = str(row.get("team1_champions", ""))
    t2_champs = str(row.get("team2_champions", ""))

    return (
        f"Metadata for the game:\n"
        f"Team 1 name: {t1_name}\n"
        f"Team 2 name: {t2_name}\n"
        f"Team 1 players: {t1_players}\n"
        f"Team 2 players: {t2_players}\n"
        f"Team 1 champions: {t1_champs}\n"
        f"Team 2 champions: {t2_champs}\n"
    )

def process_row_for_model(row: pd.Series, model: str, words_per_chunk: int, overlap_words: int) -> str:
    text = str(row.get("text", "") or "")
    if not text.strip():
        return text  # nothing to do

    metadata = build_metadata_from_row(row)
    chunks = chunk_text_by_words(text, words_per_chunk, overlap_words)

    all_pairs: List[Tuple[str, str]] = []
    for idx, chunk in enumerate(chunks):
        try:
            pairs = request_corrections_for_chunk(model, metadata, chunk)
            all_pairs.extend(pairs)
        except Exception as e:
            # Keep going even if one chunk fails
            print(f"[{model}] Chunk {idx+1}/{len(chunks)} error: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    consolidated = consolidate_corrections(all_pairs)
    fixed = apply_corrections_to_text(text, consolidated)
    return fixed

# -------------------------------
# Main
# -------------------------------
def main():
    asr_dataset = pd.read_json(INPUT_PATH, lines=True)
    asr_dataset = asr_dataset.tail(9)

    # Optional normalization you had before — only if needed for evaluation columns:
    # Ensure these columns exist before touching them; skip if they don't.
    for col in ("gpt", "fixed_text"):
        if col in asr_dataset.columns:
            asr_dataset[col] = asr_dataset[col].astype(str).str.lower()

    # For each model, create a new column {model}_newfix
    for model in MODELS:
        col_name = f"{model}_newfix"
        print(f"Processing all rows with model: {model} -> column: {col_name}")
        results = []
        for i, row in asr_dataset.iterrows():
            fixed = process_row_for_model(row, model, WORDS_PER_CHUNK, OVERLAP_WORDS)
            results.append(fixed)
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{len(asr_dataset)} rows")

        asr_dataset[col_name] = results

    # Save back to jsonl
    asr_dataset.to_json(OUTPUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()