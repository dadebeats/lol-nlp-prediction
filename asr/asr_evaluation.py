from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import pandas as pd

# Load your data
asr_dataset = pd.read_json("asr/asr_error_correction_devset.jsonl", lines=True)
asr_dataset["gpt"] = asr_dataset["gpt"].astype(str)
asr_dataset["fixed_text"] = asr_dataset["fixed_text"].astype(str)
asr_dataset["gpt"] = asr_dataset["gpt"].apply(lambda x: x.lower())
asr_dataset["fixed_text"] = asr_dataset["fixed_text"].apply(lambda x: x.lower())

# ---- Helpers ----
smooth = SmoothingFunction().method1

def tok(s: str):
    # Minimal tokenization (no punctuation dependency)
    return str(s).strip().lower().split()

def safe_bleu(ref_text: str, pred_text: str, weights):
    ref_toks = tok(ref_text)
    hyp_toks = tok(pred_text)
    if len(hyp_toks) == 0 or len(ref_toks) == 0:
        return 0.0
    try:
        return float(sentence_bleu([ref_toks], hyp_toks, weights=weights, smoothing_function=smooth))
    except ZeroDivisionError:
        return 0.0

pred_col = "gpt"
target_col = "fixed_text"
# Ensure strings (avoid NaN issues)
asr_dataset[pred_col] = asr_dataset[pred_col].fillna("").astype(str)
asr_dataset[target_col] = asr_dataset[target_col].fillna("").astype(str)

# ---- WER (reference first, hypothesis second) ----
asr_dataset["wer"] = asr_dataset.apply(
    lambda r: wer(r[target_col], r[pred_col]), axis=1
)

# ---- BLEU (token-based; punctuation not required) ----
# Add a few flavors to improve interpretability
asr_dataset["bleu1"] = asr_dataset.apply(
    lambda r: safe_bleu(r[target_col], r[pred_col], weights=(1.0, 0.0, 0.0, 0.0)), axis=1
)
asr_dataset["bleu2"] = asr_dataset.apply(
    lambda r: safe_bleu(r[target_col], r[pred_col], weights=(0.5, 0.5, 0.0, 0.0)), axis=1
)
asr_dataset["bleu4"] = asr_dataset.apply(
    lambda r: safe_bleu(r[target_col], r[pred_col], weights=(0.25, 0.25, 0.25, 0.25)), axis=1
)

# ---- BERTScore (batch for speed) ----
# Uses a contextual semantic similarity; robust to paraphrases and punctuation
preds = asr_dataset[pred_col].tolist()
refs  = asr_dataset[target_col].tolist()
P, R, F1 = bertscore(preds, refs, lang="en", rescale_with_baseline=True)

# Convert tensors to plain floats
asr_dataset["bertscore_precision"] = P.numpy()
asr_dataset["bertscore_recall"]    = R.numpy()
asr_dataset["bertscore_f1"]        = F1.numpy()

# (Optional) Save with metrics
# asr_dataset.to_json("asr_devset_with_metrics.jsonl", lines=True, orient="records", force_ascii=False)

asr_dataset.head()
