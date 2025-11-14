import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional
import json, re
from tqdm import tqdm
from link_youtube_to_oracle import team_mapping
import ast

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


def _coerce_list(x) -> List[str]:
    """
    Accepts Python-literal lists (e.g. "['Aatrox','Olaf']"), JSON lists,
    or CSV-like strings using | ; , as delimiters.
    Returns a clean list of non-empty, trimmed strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [s.strip() for s in map(str, x) if str(s).strip()]

    s = str(x).strip()
    if not s:
        return []

    # 1) Try Python literal (handles single quotes etc.)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(v).strip().strip('"').strip("'") for v in val if str(v).strip().strip('"').strip("'")]
    except Exception:
        pass

    # 2) Try JSON
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
    except Exception:
        pass

    # 3) Fallback: strip surrounding [ ] if someone dumped a stringy list
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    # 4) Split by common delimiters
    for delim in ["|", ";", ","]:
        if delim in s:
            parts = [p.strip().strip('"').strip("'") for p in s.split(delim)]
            return [p for p in parts if p]

    # 5) Single token
    return [s.strip().strip('"').strip("'")]

def _invert_team_mapping(short_to_full: Dict[str, str]) -> Dict[str, List[str]]:
    """
    {SHORT -> FULL}  -->  {FULL -> [SHORT1, SHORT2, ...]}
    """
    inv: Dict[str, List[str]] = {}
    for short, full in (short_to_full or {}).items():
        if not full:
            continue
        inv.setdefault(str(full).strip(), []).append(str(short).strip())
    return inv

def _regex_escape_map_keys(d: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    # (unchanged) — it already uses re.IGNORECASE
    items = sorted(d.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled: List[Tuple[re.Pattern, str]] = []
    for surface, repl in items:
        esc = re.escape(surface)
        pattern = re.compile(rf'(?<!\w){esc}(?!\w)', flags=re.IGNORECASE)
        compiled.append((pattern, repl))
    return compiled


@dataclass
class LoLMatchMetadata:
    team1_name: str
    team2_name: str
    team1_players: List[str]
    team2_players: List[str]
    team1_champions: List[str]
    team2_champions: List[str]

    @classmethod
    def from_row(cls, row: pd.Series) -> "LoLMatchMetadata":
        t1 = str(row.get("team1_name", "") or "").strip()
        t2 = str(row.get("team2_name", "") or "").strip()
        t1p = _coerce_list(row.get("team1_players", []))
        t2p = _coerce_list(row.get("team2_players", []))
        t1c = _coerce_list(row.get("team1_champions", []))
        t2c = _coerce_list(row.get("team2_champions", []))
        return cls(
            team1_name=t1, team2_name=t2,
            team1_players=t1p, team2_players=t2p,
            team1_champions=t1c, team2_champions=t2c
        )

    def to_string(self) -> str:
        return (
            "Metadata for the game:\n"
            f"Team 1 name: {self.team1_name}\n"
            f"Team 2 name: {self.team2_name}\n"
            f"Team 1 players: {', '.join(self.team1_players)}\n"
            f"Team 2 players: {', '.join(self.team2_players)}\n"
            f"Team 1 champions: {', '.join(self.team1_champions)}\n"
            f"Team 2 champions: {', '.join(self.team2_champions)}\n"
        )

    def build_replacements(self) -> Dict[str, str]:
        """
        Build surface-form -> placeholder map.
        Includes team names, team shortcuts (from team_mapping, inverted),
        players, and champions. Matching is case-insensitive at replacement time.
        """
        mapping: Dict[str, str] = {}

        # Invert imported team_mapping: {SHORT -> FULL} -> {FULL -> [SHORT...]}
        try:
            inv_map = _invert_team_mapping(team_mapping)
        except Exception:
            inv_map = {}

        # Teams (full names)
        if self.team1_name:
            mapping[self.team1_name] = "<TEAM1>"
            # add shortcuts for team1
            for sh in inv_map.get(self.team1_name, []):
                mapping[sh] = "<TEAM1>"
        if self.team2_name:
            mapping[self.team2_name] = "<TEAM2>"
            # add shortcuts for team2
            for sh in inv_map.get(self.team2_name, []):
                mapping[sh] = "<TEAM2>"

        # Players
        for i, p in enumerate(self.team1_players, start=1):
            if p:
                mapping[p] = f"<T1_PLAYER_{i}>"
        for i, p in enumerate(self.team2_players, start=1):
            if p:
                mapping[p] = f"<T2_PLAYER_{i}>"

        # Champions
        for i, c in enumerate(self.team1_champions, start=1):
            if c:
                mapping[c] = f"<T1_CHAMPION_{i}>"
        for i, c in enumerate(self.team2_champions, start=1):
            if c:
                mapping[c] = f"<T2_CHAMPION_{i}>"

        return mapping

    def mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Returns masked text + the surface->placeholder dict used (for debugging/audits).
        Replacement is case-insensitive and prefers longer matches first.
        """
        mapping = self.build_replacements()
        patterns = _regex_escape_map_keys(mapping)

        def _sub_once(s: str) -> str:
            for pat, repl in patterns:
                s = pat.sub(repl, s)
            return s

        masked = _sub_once(text)
        return masked, mapping

def mask_entities(text: str, metadata: LoLMatchMetadata) -> str:
    masked, _ = metadata.mask_text(text)
    return masked


# ------------------------
# Main: take a DataFrame with 'text' column and add 'embedding'
# ------------------------
def add_embeddings_to_df(
    df: pd.DataFrame,
    text_col: str = "text",
    mask_entities: bool = False,  # kept the name for backward compat
) -> pd.DataFrame:
    assert text_col in df.columns, f"DataFrame missing '{text_col}' column"

    # avoid name shadowing: grab the global masking function once
    mask_entities_fn = globals().get("mask_entities")

    out = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunk+Embed per row"):
        text = str(row[text_col])
        if mask_entities:
            # build metadata per row and mask
            meta = LoLMatchMetadata.from_row(row)
            # call the real function (not the boolean param)
            text = mask_entities_fn(text, meta) if callable(mask_entities_fn) else text

        out.append(embed_with_chunking(text))

    df = df.copy()
    col = "embedding_masked" if mask_entities else "embedding"
    df[col] = [v.astype(np.float32).tolist() for v in out]
    return df

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Example DataFrame
    df = pd.read_csv("dataset.csv", index_col=[0])
    use_masking_in_text = True
    # Pick the correct column name
    emb_col = "embedding_masked" if use_masking_in_text else "embedding"
    df_with_vecs = add_embeddings_to_df(df, text_col="text", mask_entities=True)
    df_with_vecs = add_embeddings_to_df(df_with_vecs, text_col="text", mask_entities=False)
    df_with_vecs.to_parquet(f"dataset.parquet")
    print("Saved: dataset.parquet")
