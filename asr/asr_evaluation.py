from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import pandas as pd

# ---- Load ----
asr_dataset = pd.read_json("asr/asr_error_correction_devset_corrected.jsonl", lines=True)
asr_dataset = asr_dataset.fillna("").astype(str)
asr_dataset = asr_dataset.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# ---- Config ----
gold_col = "fixed_text"
model_cols = [
    "gpt", "gpt-5_newfix", "gpt-5-mini_newfix",
    "gpt-4o-mini_newfix", "o3_newfix", "gpt-5-nano_newfix",
    "gpt-4.1_newfix"
]
smooth = SmoothingFunction().method1

def tok(s: str): return s.strip().split()

def safe_bleu(ref_text: str, pred_text: str):
    ref_toks, hyp_toks = tok(ref_text), tok(pred_text)
    if not ref_toks or not hyp_toks:
        return 0.0
    return float(sentence_bleu([ref_toks], hyp_toks, weights=(1.0, 0, 0, 0), smoothing_function=smooth))

# ---- Compute metrics ----
results = []
for model_col in model_cols:
    preds = asr_dataset[model_col].tolist()
    refs  = asr_dataset[gold_col].tolist()

    # --- BLEU (simple unigram BLEU-1) ---
    bleu_scores = [safe_bleu(r, p) for r, p in zip(refs, preds)]

    # --- WER ---
    wer_scores = [wer(r, p) for r, p in zip(refs, preds)]

    # --- BERTScore ---
    P, R, F1 = bertscore(preds, refs, lang="en", rescale_with_baseline=True)
    bert_f1 = F1.numpy().tolist()

    # --- Aggregate ---
    results.append({
        "model": model_col,
        "BLEU1_mean": sum(bleu_scores) / len(bleu_scores),
        "WER_mean": sum(wer_scores) / len(wer_scores),
        "BERTScore_F1_mean": sum(bert_f1) / len(bert_f1)
    })

# ---- Report ----
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("BERTScore_F1_mean", ascending=False)


# ---- (Optional) Save ----
metrics_df.to_csv("asr_model_comparison.csv", index=False)
