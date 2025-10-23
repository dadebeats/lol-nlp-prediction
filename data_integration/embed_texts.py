# --- Setup (one-time) ---
# pip install -U "sentence-transformers>=3.0.0" transformers torch pandas tqdm

import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# ------------------------
# Config
# ------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
BATCH_SIZE = 64
CHUNK_TOKENS = 512       # per-chunk token cap (<= model max length; 384 is a good sweet spot)
CHUNK_OVERLAP = 64       # token overlap for context continuity
NORMALIZE = True         # cosine-friendly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load model + tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# For some models, tokenizer.model_max_length might be very large (int(1e30)); cap it manually:
MODEL_MAX_LEN = min(
    getattr(tokenizer, "model_max_length", CHUNK_TOKENS),
    8192  # hard safety cap
)
CHUNK_TOKENS = min(CHUNK_TOKENS, MODEL_MAX_LEN)

# ------------------------
# Chunking by tokens
# ------------------------
def chunk_by_tokens(text: str, max_tokens: int, overlap: int) -> Tuple[List[str], List[int]]:
    """
    Returns:
      chunks: list[str] of text chunks
      lengths: list[int] token lengths per chunk (for weighted averaging)
    """
    if not text:
        return [""], [0]

    # Tokenize once
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]

    if len(ids) <= max_tokens:
        return [text], [len(ids)]

    # Slide over token ids with overlap
    step = max_tokens - overlap
    chunks, lengths = [], []
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        sub_ids = ids[start:end]
        if not sub_ids:
            break
        # Decode sub-ids back to string chunk
        chunk_text = tokenizer.decode(sub_ids, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        lengths.append(len(sub_ids))
        if end == len(ids):
            break
    return chunks, lengths

# ------------------------
# Embedding helpers
# ------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns np.ndarray of shape (len(texts), dim)
    """
    # SentenceTransformer handles batching internally if needed,
    # but manual batching lets us show progress and be explicit.
    vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False
        )
        vecs.append(emb)
    return np.vstack(vecs) if vecs else np.zeros((0,))

def embed_with_chunking(text: str) -> np.ndarray:
    """
    Chunk a single long text by tokens, embed each chunk,
    then return a token-length–weighted average embedding.
    """
    chunks, lengths = chunk_by_tokens(text, CHUNK_TOKENS, CHUNK_OVERLAP)
    embs = embed_texts(chunks)  # (num_chunks, dim)
    if embs.ndim == 1:
        return embs

    # Weighted average by token length to avoid biasing short chunks
    weights = np.array(lengths, dtype=np.float32)
    weights = np.maximum(weights, 1.0)
    weights = weights / weights.sum()
    pooled = (embs * weights[:, None]).sum(axis=0)

    # If we normalized chunk embeddings, re-normalize the pooled vector to unit length
    if NORMALIZE:
        norm = np.linalg.norm(pooled) + 1e-12
        pooled = pooled / norm
    return pooled

# ------------------------
# Main: take a DataFrame with 'text' column and add 'embedding'
# ------------------------
def add_embeddings_to_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    assert text_col in df.columns, f"DataFrame missing '{text_col}' column"
    out = []
    for t in tqdm(df[text_col].tolist(), desc="Chunk+Embed per row"):
        out.append(embed_with_chunking(str(t)))
    # Store as Python lists (JSON-serializable) or keep as np.ndarray separately
    df = df.copy()
    df["embedding"] = [v.astype(np.float32).tolist() for v in out]
    return df

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Example DataFrame
    df = pd.read_csv("dataset.csv", index_col="gameid")

    df_with_vecs = add_embeddings_to_df(df, text_col="text")

    # Save to Parquet (keeps lists fine) or NPY
    df_with_vecs.to_parquet("transcripts_with_embeddings.parquet", index=False)
    np.save("embeddings.npy", np.vstack([np.array(v, dtype=np.float32) for v in df_with_vecs["embedding"]]))

    print("Vector dim:", len(df_with_vecs["embedding"].iloc[0]))
    print("Saved: transcripts_with_embeddings.parquet and embeddings.npy")
