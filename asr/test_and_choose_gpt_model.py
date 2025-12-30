# This script runs a batch of ASR error correction requests against OpenAI's GPT-5 nano
# Saves a DataFrame with the results.'
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

MODELS = [
    "gpt-5-nano",
]
WORDS_PER_CHUNK = 250
OVERLAP_WORDS = 10
REQUEST_DELAY_SEC = 0.2

INPUT_PATH = "asr/asr_error_correction_devset.jsonl"
OUTPUT_PATH = "asr/asr_error_correction_devset_corrected.jsonl"


def chunk_text_by_words(
    text: str,
    words_per_chunk: int = 250,
    overlap_words: int = 25
) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    n = len(words)
    if n == 0:
        return []

    chunks = []
    start = 0
    while start < n:
        end = min(start + words_per_chunk, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def parse_corrections(raw: str) -> List[Tuple[str, str]]:
    """
    Parse correction pairs from model output.
    Supports JSON {"corrections":[{"from":...,"to":...}]} or line-based "from, to".
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("corrections"), list):
            pairs = []
            for item in data["corrections"]:
                if isinstance(item, dict):
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


def consolidate_corrections(
    corrections: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Resolve multiple corrections for the same source string by majority vote.
    Ties are broken by last occurrence. Longer source strings are applied first.
    """
    by_from = defaultdict(list)
    for frm, to in corrections:
        if frm and to and frm != to:
            by_from[frm].append(to)

    final = []
    for frm, tos in by_from.items():
        counts = Counter(tos)
        max_freq = max(counts.values())
        candidates = {t for t, c in counts.items() if c == max_freq}
        for t in reversed(tos):
            if t in candidates:
                final.append((frm, t))
                break

    final.sort(key=lambda p: len(p[0]), reverse=True)
    return final


def is_token_like(s: str) -> bool:
    """Heuristic for determining whether a string should be replaced as a token."""
    return bool(re.fullmatch(r"[A-Za-z0-9'\- ]+", s))


def replace_one_case_insensitive(text: str, frm: str, to: str) -> str:
    """Case-insensitive replacement with token-boundary preference."""
    escaped = re.escape(frm)
    if is_token_like(frm):
        pattern = r"(?i)(?<!\w)" + escaped + r"(?!\w)"
    else:
        pattern = r"(?i)" + escaped

    new_text, count = re.subn(pattern, to, text)
    if count == 0 and is_token_like(frm):
        new_text = re.sub(r"(?i)" + escaped, to, text)
    return new_text


def apply_corrections_to_text(
    text: str,
    corrections: List[Tuple[str, str]]
) -> str:
    """Apply ordered string corrections to text."""
    fixed = text
    for frm, to in corrections:
        if frm and to and frm != to:
            fixed = replace_one_case_insensitive(fixed, frm, to)
    return fixed


def request_corrections_for_chunk(
    model: str,
    metadata: str,
    chunk: str
) -> List[Tuple[str, str]]:
    """Request ASR corrections for a single transcript chunk."""
    prompt = (
        "Check the provided transcript CHUNK (not the full game) for ASR errors. "
        "Use only the metadata to ground entity corrections. "
        "Return ONLY the proposed corrections, no explanations. "
        "Format: one correction per line as 'incorrect string, corrected string'. "
        "Only include corrections that appear in THIS CHUNK.\n\n"
        f"{metadata}\n\n"
        f"CHUNK START\n{chunk}\nCHUNK END"
    )

    response = client.responses.create(
        model=model,
        instructions=(
            "You are an ASR post-processor for League of Legends commentary transcripts. "
            "Output only correction pairs in the required format."
        ),
        input=prompt,
    )

    return parse_corrections(response.output_text)


def build_metadata_from_row(row: pd.Series) -> str:
    """Serialize entity metadata used to ground ASR corrections."""
    return (
        f"Metadata for the game:\n"
        f"Team 1 name: {row.get('team1_name', '')}\n"
        f"Team 2 name: {row.get('team2_name', '')}\n"
        f"Team 1 players: {row.get('team1_players', '')}\n"
        f"Team 2 players: {row.get('team2_players', '')}\n"
        f"Team 1 champions: {row.get('team1_champions', '')}\n"
        f"Team 2 champions: {row.get('team2_champions', '')}\n"
    )


def process_row_for_model(
    row: pd.Series,
    model: str,
    words_per_chunk: int,
    overlap_words: int
) -> str:
    """Apply chunked ASR correction to a single transcript row."""
    text = str(row.get("text", "") or "")
    if not text.strip():
        return text

    metadata = build_metadata_from_row(row)
    chunks = chunk_text_by_words(text, words_per_chunk, overlap_words)

    all_pairs: List[Tuple[str, str]] = []
    for idx, chunk in enumerate(chunks):
        try:
            all_pairs.extend(
                request_corrections_for_chunk(model, metadata, chunk)
            )
        except Exception as e:
            print(f"[{model}] Chunk {idx+1}/{len(chunks)} error: {e}")
        time.sleep(REQUEST_DELAY_SEC)

    consolidated = consolidate_corrections(all_pairs)
    return apply_corrections_to_text(text, consolidated)

if __name__ == "__main__":
    asr_dataset = pd.read_json(INPUT_PATH, lines=True)
    asr_dataset = asr_dataset.tail(9)

    for col in ("gpt", "fixed_text"):
        if col in asr_dataset.columns:
            asr_dataset[col] = asr_dataset[col].astype(str).str.lower()

    for model in MODELS:
        col_name = f"{model}_newfix"
        results = []
        for _, row in asr_dataset.iterrows():
            results.append(
                process_row_for_model(
                    row, model, WORDS_PER_CHUNK, OVERLAP_WORDS
                )
            )
        asr_dataset[col_name] = results

    asr_dataset.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False
    )
