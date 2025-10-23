import json
import pandas as pd

asr_dataset = pd.read_json("asr_error_correction_devset.jsonl", lines=True)

def _fmt_list(x):
    # Accept list or string; normalize to comma-separated string
    if isinstance(x, list):
        return ", ".join(map(str, x))
    return str(x) if pd.notna(x) else ""

def build_prompt(row) -> str:
    t1_name = row.get("team1_name", "")
    t1_players = _fmt_list(row.get("team1_players", ""))
    t1_champs  = _fmt_list(row.get("team1_champions", ""))

    t2_name = row.get("team2_name", "")
    t2_players = _fmt_list(row.get("team2_players", ""))
    t2_champs  = _fmt_list(row.get("team2_champions", ""))

    text = str(row.get("text", ""))

    prompt = (
        "You are an ASR correction system for League of Legends live commentary.\n\n"
        "Metadata:\n"
        f"- Team 1: {t1_name}\n"
        f"  • Players: {t1_players}\n"
        f"  • Champions: {t1_champs}\n"
        f"- Team 2: {t2_name}\n"
        f"  • Players: {t2_players}\n"
        f"  • Champions: {t2_champs}\n\n"
        "Instructions:\n"
        "- For these metadata (teams, players, champions), act as an ASR correction system for League of Legends commentaries.\n"
        "- For text passages that are likely incorrect, pick the most probable replacement using common English and the metadata.\n"
        "- Do NOT paraphrase; do word-by-word/minimal edits (spelling, casing, spacing, punctuation, named entities).\n"
        "- The text comes from the jsonlines file so it's one long string without line breaks, so preserve that so it can be easily copy pasted to the json file.\n"
        "- Output ONLY the corrected text — no explanations.\n\n"
        "Text:\n"
        f"{text}"
    )
    return prompt

# Create a new column with the prompt
asr_dataset["gpt_prompt"] = asr_dataset.apply(build_prompt, axis=1)

# (Optional) Save prompts as JSONL with vid_name for traceability
# Each line: {"vid_name": "...", "prompt": "..."}
with open("asr_correction_prompts.jsonl", "w", encoding="utf-8") as f:
    for _, r in asr_dataset.iterrows():
        f.write(json.dumps({
            "vid_name": r.get("vid_name", ""),
            "prompt": r["gpt_prompt"]
        }, ensure_ascii=False) + "\n")

# Quick peek
print(asr_dataset[["vid_name", "gpt_prompt"]].head(2).to_string(index=False))
