import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json, re
from tqdm import tqdm
from link_youtube_to_oracle import team_mapping
import ast
from openai import OpenAI
from text_and_embedding_config import TextAndEmbeddingConfig, EmbeddingModelConfig, ChunkingConfig

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 64
NORMALIZE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class EmbeddingFactory:
    def __init__(self, config: TextAndEmbeddingConfig):
        self.config = config
        model_name = config.model.model_name
        masking = set(config.masking_types)

        self.mask_team = ("team" in masking)
        self.mask_player = ("player" in masking)
        self.mask_champion = ("champion" in masking)

        huggingface_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        openai_models = ["text-embedding-3-large"]

        self.NORMALIZE = True

        if model_name in huggingface_models:
            self.mode = "huggingface"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = SentenceTransformer(model_name, device=DEVICE)
            self.set_chunk_config(
                chunk_tokens=config.chunking.max_tokens if config.chunking else 512,
                chunk_overlap=config.chunking.overlap if config.chunking else 256,
            )
        elif model_name in openai_models:
            self.mode = "openai"
            self.model = config.model.model_name
            self.client = OpenAI()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    # masking entry point
    def mask_text(self, row: pd.Series, text: str) -> str:
        meta = LoLMatchMetadata.from_row(row)

        mapping = {}

        if self.mask_team:
            mapping.update(meta._team_map())
        if self.mask_player:
            mapping.update(meta._player_map())
        if self.mask_champion:
            mapping.update(meta._champion_map())

        return self._apply_mapping(text, mapping)

    @staticmethod
    def _apply_mapping(text: str, mapping: Dict[str, str]) -> str:
        items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        for surface, repl in items:
            pat = re.compile(rf'(?<!\w){re.escape(surface)}(?!\w)', flags=re.IGNORECASE)
            text = pat.sub(repl, text)
        return text

    def set_chunk_config(self, chunk_tokens, chunk_overlap):
        # For some models, tokenizer.model_max_length might be very large (int(1e30)); cap it manually:
        model_max_len = min(
            getattr(self.tokenizer, "model_max_length", chunk_tokens),
            8192  # hard safety cap
        )
        self.CHUNK_TOKENS = min(chunk_tokens, model_max_len)
        self.CHUNK_OVERLAP = chunk_overlap


    def _chunk_by_tokens_hf(self, text: str):
        """
        Returns:
          chunks: list[str] of text chunks
          lengths: list[int] token lengths per chunk (for weighted averaging)
        """
        if not text:
            return [""], [0]

        # Tokenize once
        enc = self.tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]

        if len(ids) <= self.CHUNK_TOKENS:
            return [text], [len(ids)]

        # Slide over token ids with overlap
        step = self.CHUNK_TOKENS - self.CHUNK_OVERLAP
        chunks, lengths = [], []
        for start in range(0, len(ids), step):
            end = min(start + self.CHUNK_TOKENS, len(ids))
            sub_ids = ids[start:end]
            if not sub_ids:
                break
            # Decode sub-ids back to string chunk
            chunk_text = self.tokenizer.decode(sub_ids, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            lengths.append(len(sub_ids))
            if end == len(ids):
                break
        return chunks, lengths

    def _embed_texts_hf(self, texts: List[str]) -> np.ndarray:
        """
        Returns np.ndarray of shape (len(texts), dim)
        """
        # SentenceTransformer handles batching internally if needed,
        # but manual batching lets us show progress and be explicit.
        vecs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i:i + BATCH_SIZE]
            emb = self.model.encode(
                batch,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=self.NORMALIZE,
                show_progress_bar=False
            )
            vecs.append(emb)
        return np.vstack(vecs) if vecs else np.zeros((0,))

    @staticmethod
    def _chunk_by_tokens_openai(text: str, chunk_size: int = 28000):
        """
        Split text into chunks of at most `chunk_size` characters.
        Returns:
            chunks:  list[str]
            lengths: list[int]
        """
        if not text:
            return [""], [0]

        chunks = []
        lengths = []

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
            lengths.append(len(chunk))

        return chunks, lengths


    def _embed_texts_openai(self, texts: List[str]) -> np.ndarray:
        """
        Returns a numpy array embedding for the input text.
        OpenAI handles tokenization internally.
        """
        vecs = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            # Extract vector directly
            emb = response.data[0].embedding
            vecs.append(emb)
        return np.vstack(vecs) if vecs else np.zeros((0,))


    def _chunk_by_tokens(self, text):
        if self.mode == "huggingface":
            return self._chunk_by_tokens_hf(text)
        elif self.mode == "openai":
            return self._chunk_by_tokens_openai(text)
        else:
            raise ValueError(f"Unknown model mode: {self.mode}")

    def _embed_texts(self, texts):
        if self.mode == "huggingface":
            return self._embed_texts_hf(texts)
        elif self.mode == "openai":
            return self._embed_texts_openai(texts)
        else:
            raise ValueError(f"Unknown model mode: {self.mode}")

    def embed_with_chunking(self, text: str) -> np.ndarray:
        """
        Chunk a single long text by tokens, embed each chunk,
        then return a token-length–weighted average embedding.
        """
        chunks, lengths = self._chunk_by_tokens(text)
        embs = self._embed_texts(chunks)  # (num_chunks, dim)
        if embs.ndim == 1:
            return embs

        # Weighted average by token length to avoid biasing short chunks
        weights = np.array(lengths, dtype=np.float32)
        weights = np.maximum(weights, 1.0)
        weights = weights / weights.sum()
        pooled = (embs * weights[:, None]).sum(axis=0)

        # If we normalized chunk embeddings, re-normalize the pooled vector to unit length
        if self.NORMALIZE:
            norm = np.linalg.norm(pooled) + 1e-12
            pooled = pooled / norm
        return pooled

# ------------------------
# Embedding helpers
# ------------------------

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
        return cls(
            str(row.get("team1_name","")).strip(),
            str(row.get("team2_name","")).strip(),
            _coerce_list(row.get("team1_players", [])),
            _coerce_list(row.get("team2_players", [])),
            _coerce_list(row.get("team1_champions", [])),
            _coerce_list(row.get("team2_champions", [])),
        )

    # team masking only
    def _team_map(self) -> Dict[str,str]:
        mapping = {}
        inv = _invert_team_mapping(team_mapping)
        if self.team1_name:
            mapping[self.team1_name] = "<TEAM1>"
            for sh in inv.get(self.team1_name, []):
                mapping[sh] = "<TEAM1>"
        if self.team2_name:
            mapping[self.team2_name] = "<TEAM2>"
            for sh in inv.get(self.team2_name, []):
                mapping[sh] = "<TEAM2>"
        return mapping

    # player masking only
    def _player_map(self) -> Dict[str,str]:
        mapping = {}
        for i,p in enumerate(self.team1_players, start=1):
            mapping[p] = f"<T1_PLAYER_{i}>"
        for i,p in enumerate(self.team2_players, start=1):
            mapping[p] = f"<T2_PLAYER_{i}>"
        return mapping

    # champion masking only
    def _champion_map(self) -> Dict[str,str]:
        mapping = {}
        for i,c in enumerate(self.team1_champions, start=1):
            mapping[c] = f"<T1_CHAMPION_{i}>"
        for i,c in enumerate(self.team2_champions, start=1):
            mapping[c] = f"<T2_CHAMPION_{i}>"
        return mapping

def mask_entities(text: str, metadata: LoLMatchMetadata) -> str:
    masked, _ = metadata.mask_text(text)
    return masked


# ------------------------
# Main: take a DataFrame with 'text' column and add 'embedding'
# ------------------------
def add_embeddings_to_df(
    df: pd.DataFrame,
    ef: EmbeddingFactory,
    text_col: str = "text",
) -> pd.DataFrame:

    out_embs, out_masked = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunk+Embed per row"):
        text = str(row[text_col])

        # get masked version based on config
        masked = text
        if ef.mask_team or ef.mask_player or ef.mask_champion:
            masked = ef.mask_text(row, text)

        out_masked.append(masked)
        out_embs.append(ef.embed_with_chunking(masked))

    df = df.copy()
    df["masked_text"] = out_masked
    df["embedding"] = [v.astype(np.float32).tolist() for v in out_embs]
    return df

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    chunking_config = ChunkingConfig(
        max_tokens=512,
        overlap=0
    )
    config = TextAndEmbeddingConfig(
        masking_types=("player", "team"),
        asr="corrected",
        model=EmbeddingModelConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            shortcut="oai_emb3"
        ),
        chunking=None
    )
    embedding_factory = EmbeddingFactory(config)

    df = pd.read_csv("dataset.csv", index_col=[0])
    df_with_vecs = add_embeddings_to_df(df, embedding_factory, text_col="text")
    df_with_vecs = add_embeddings_to_df(df_with_vecs, embedding_factory, text_col="text")
    df_with_vecs.to_parquet(f"{config.to_string()}.parquet")
    print(f"{config.to_string()}.parquet")
