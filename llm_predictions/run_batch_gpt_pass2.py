"""
PASS 2: Predict the target match winner from sequential event histories.

Input records are produced by `build_pass2_sequence_jsonl.py` and contain:
- a target match (team1/team2 + roster metadata),
- k-match histories for each team (POV-aligned outcomes + event lists).

This script renders one prompt per record, executes batch inference via the
OpenAI Responses API, and writes one JSONL line per record containing:
- predicted label (0/1),
- raw model output,
- selected target metadata for downstream evaluation.
"""

import os
import time
import json
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-4.1"

INPUT_JSONL = "sequence_pass2.jsonl"
OUTPUT_JSONL = "pass2_predictions.jsonl"

BATCH_INPUT_TEMPLATE = "batch_input_pass2_{idx:03d}.jsonl"
BATCH_OUTPUT_TEMPLATE = "batch_output_pass2_{idx:03d}.jsonl"

POLL_INTERVAL_SEC = 30

MAX_TOKENS_PER_BATCH = 900_000
CHARS_PER_TOKEN = 4.0
PROMPT_OVERHEAD_TOKENS = 250

DIGIT_RE = re.compile(r"[01]")


def _stringify_players(players: Any) -> str:
    """Render roster/player lists into a stable comma-separated string."""
    if isinstance(players, (list, tuple)):
        return ", ".join(map(str, players))
    return str(players or "")


def _format_events(events: List[Dict[str, Any]]) -> str:
    """Render event lists as labeled bullet lines."""
    out = []
    for ev in events or []:
        lbl = str(ev.get("label", "") or "").strip()
        txt = str(ev.get("text", "") or "").strip()
        if not txt:
            continue
        out.append(f"- [{lbl}] {txt}" if lbl else f"- {txt}")
    return "\n".join(out)


def build_pass2_prompt_from_record(rec: Dict[str, Any]) -> str:
    """
    Render a deterministic prompt from a Pass 2 record.

    Format:
    - target match header (teams + player rosters),
    - team1 history (oldest -> newest),
    - team2 history (oldest -> newest),
    - instruction to output a single digit: 1 if team1 wins, else 0.
    """
    t = rec["target"]
    h = rec["history"]

    team1 = str(t.get("team1_name", "") or "")
    team2 = str(t.get("team2_name", "") or "")

    team1_players = _stringify_players(t.get("team1_players", ""))
    team2_players = _stringify_players(t.get("team2_players", ""))

    lines = []
    lines.append("Predict the winner of the TARGET match.")
    lines.append("Output exactly one character: '1' if team1 wins, else '0'.")
    lines.append("No other text.")
    lines.append("")
    lines.append("TARGET MATCH:")
    lines.append(f"- team1: {team1}")
    lines.append(f"- team1 players: {team1_players}")
    lines.append(f"- team2: {team2}")
    lines.append(f"- team2 players: {team2_players}")
    lines.append("")
    lines.append("History note: pov_win=1 indicates a win for the listed POV team; pov_win=0 indicates a loss.")
    lines.append("")

    lines.append(f"TEAM1 HISTORY (oldest -> newest): {team1}")
    for m in h.get("team1", []):
        opp = str(m.get("opponent", "") or "")
        mdate = str(m.get("date", "") or "")
        pov_win = int(m.get("pov_win", 0))
        outcome = "WIN" if pov_win == 1 else "LOSS"
        lines.append(f"Match vs {opp} on {mdate} | outcome={outcome} (pov_win={pov_win})")
        ev_txt = _format_events(m.get("events", []))
        lines.append(ev_txt if ev_txt else "- (no events)")
        lines.append("")

    lines.append(f"TEAM2 HISTORY (oldest -> newest): {team2}")
    for m in h.get("team2", []):
        opp = str(m.get("opponent", "") or "")
        mdate = str(m.get("date", "") or "")
        pov_win = int(m.get("pov_win", 0))
        outcome = "WIN" if pov_win == 1 else "LOSS"
        lines.append(f"Match vs {opp} on {mdate} | outcome={outcome} (pov_win={pov_win})")
        ev_txt = _format_events(m.get("events", []))
        lines.append(ev_txt if ev_txt else "- (no events)")
        lines.append("")

    lines.append("Return only: 1 if team1 wins, else 0.")
    return "\n".join(lines)


def approximate_tokens_for_prompt(prompt: str) -> int:
    """Heuristic token estimate used for batch budgeting."""
    return int(len(prompt or "") / CHARS_PER_TOKEN) + PROMPT_OVERHEAD_TOKENS


def load_sequence_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load one JSON object per line."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_records_into_batches(records: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Group record indices into token-budgeted batches based on rendered prompt size.
    """
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_tokens = 0

    for idx, rec in enumerate(records):
        row_tokens = approximate_tokens_for_prompt(build_pass2_prompt_from_record(rec))

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


def create_batch_input_file(records: List[Dict[str, Any]], rec_indices: List[int], jsonl_path: str) -> None:
    """Write one Responses API task per record into a batch JSONL file."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx in rec_indices:
            body = {
                "model": MODEL,
                "instructions": "Return only one character: '1' if team1 wins, else '0'.",
                "input": build_pass2_prompt_from_record(records[idx]),
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
    Execute one batch job: upload input JSONL, create batch, poll, download output JSONL.
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
        if getattr(batch, "error_file_id", None):
            try:
                err_bytes = client.files.content(batch.error_file_id).content
                print(err_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                pass
        raise RuntimeError(f"Batch {batch_idx} did not complete successfully (status={batch.status}).")

    out_path = BATCH_OUTPUT_TEMPLATE.format(idx=batch_idx)
    result_bytes = client.files.content(batch.output_file_id).content
    with open(out_path, "wb") as f:
        f.write(result_bytes)

    return out_path


def _extract_output_text_from_batch_line(obj: Dict[str, Any]) -> str:
    """Extract text output from a batch result line across likely response shapes."""
    body = obj.get("response", {}).get("body", {}) or {}
    output = body.get("output")

    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        return c["text"]

    try:
        return body["output"][1]["content"][0]["text"]
    except Exception:
        return ""


def parse_pass2_pred(raw: str) -> int:
    """
    Parse a binary prediction from the model output.

    Expected output is a single character "0" or "1". If additional text is
    present, the first occurrence of a 0/1 digit is used. Returns -1 if no
    valid digit is found.
    """
    raw = (raw or "").strip()
    if raw in ("0", "1"):
        return int(raw)
    m = DIGIT_RE.search(raw)
    return int(m.group(0)) if m else -1


def load_batch_results(results_jsonl_path: str) -> Dict[int, str]:
    """Load batch output JSONL into a mapping {record_idx: raw_output_text}."""
    mapping: Dict[int, str] = {}
    with open(results_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            try:
                rec_idx = int(cid.split("row-")[-1])
            except Exception:
                continue
            mapping[rec_idx] = _extract_output_text_from_batch_line(obj)
    return mapping


def load_batches(start_id: int, length: int) -> Dict[int, str]:
    """Load a contiguous range of already-downloaded batch output JSONL files."""
    all_raw_results: Dict[int, str] = {}
    for i in range(start_id, start_id + length):
        path = BATCH_OUTPUT_TEMPLATE.format(idx=i)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing batch output file: {path}")
        all_raw_results |= load_batch_results(path)
    return all_raw_results


if __name__ == "__main__":

    records = load_sequence_jsonl(INPUT_JSONL)
    batches = split_records_into_batches(records)

    all_raw_results: Dict[int, str] = {}

    for batch_idx, rec_indices in enumerate(batches, start=9):
        input_path = BATCH_INPUT_TEMPLATE.format(idx=batch_idx)
        create_batch_input_file(records, rec_indices, input_path)
        result_path = run_single_batch(input_path, batch_idx)
        all_raw_results.update(load_batch_results(result_path))

    if not all_raw_results:
        raise ValueError("No batch outputs collected.")

    n_ok = 0
    n_bad = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(records):
            raw = all_raw_results.get(idx, "")
            pred = parse_pass2_pred(raw)

            t = rec.get("target", {})
            out_obj = {
                "row_idx": idx,
                "match_idx": t.get("match_idx"),
                "date": t.get("date"),
                "team1_name": t.get("team1_name"),
                "team2_name": t.get("team2_name"),
                "team1_result": t.get("team1_result"),
                "pred": pred,
                "raw_text": raw,
            }

            if pred in (0, 1):
                n_ok += 1
            else:
                n_bad += 1

            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Pass2 parsing summary: ok={n_ok}, bad_or_empty={n_bad}")
    print(f"Saved pass2 predictions to {OUTPUT_JSONL}")
