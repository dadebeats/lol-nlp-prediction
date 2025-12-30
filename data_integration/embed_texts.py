from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from link_youtube_to_oracle import team_mapping
from text_and_embedding_config import ChunkingConfig, EmbeddingModelConfig, TextAndEmbeddingConfig

load_dotenv()

BATCH_SIZE: int = 64
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingFactory:
    """Factory for masking match entities and producing pooled text embeddings.

    Supports:
    - HuggingFace SentenceTransformers models (via `SentenceTransformer`)
    - OpenAI embedding models (via `OpenAI` client)

    Long texts are chunked, embedded per chunk, then pooled (mean-weighted or max).
    """

    def __init__(self, config: TextAndEmbeddingConfig) -> None:
        self.config = config
        model_name: str = config.model.model_name
        masking = set(config.masking_types)

        self.mask_team: bool = "team" in masking
        self.mask_player: bool = "player" in masking
        self.mask_champion: bool = "champion" in masking

        # Pooling config (default = "mean")
        self.pooling: str = getattr(config, "pooling", "mean")
        if self.pooling not in ("mean", "max"):
            raise ValueError(f"Unsupported pooling={self.pooling}")

        huggingface_models = {
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        }
        openai_models = {"text-embedding-3-large"}

        self.NORMALIZE: bool = True

        if model_name in huggingface_models:
            self.mode: str = "huggingface"
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

    def mask_text(self, row: pd.Series, text: str) -> str:
        """Mask team/player/champion surface forms in `text` using metadata from `row`."""
        meta = LoLMatchMetadata.from_row(row)
        mapping: Dict[str, str] = {}
        if self.mask_team:
            mapping.update(meta._team_map())
        if self.mask_player:
            mapping.update(meta._player_map())
        if self.mask_champion:
            mapping.update(meta._champion_map())
        return self._apply_mapping(text, mapping)

    @staticmethod
    def _apply_mapping(text: str, mapping: Dict[str, str]) -> str:
        """Apply replacements using whole-word, case-insensitive matching (longest keys first)."""
        items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        for surface, repl in items:
            pat = re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", flags=re.IGNORECASE)
            text = pat.sub(repl, text)
        return text

    def set_chunk_config(self, chunk_tokens: int, chunk_overlap: int) -> None:
        """Configure token-based chunking for HF models, respecting the tokenizer's max length."""
        model_max_len = min(getattr(self.tokenizer, "model_max_length", chunk_tokens), 8192)
        self.CHUNK_TOKENS: int = min(chunk_tokens, model_max_len)
        self.CHUNK_OVERLAP: int = chunk_overlap

    def _chunk_by_tokens_hf(self, text: str) -> tuple[list[str], list[int]]:
        """Chunk text by tokenizer tokens for HuggingFace models."""
        if not text:
            return [""], [0]
        enc = self.tokenizer(text, add_special_tokens=False)
        ids: list[int] = enc["input_ids"]

        if len(ids) <= self.CHUNK_TOKENS:
            return [text], [len(ids)]

        step = self.CHUNK_TOKENS - self.CHUNK_OVERLAP
        chunks: list[str] = []
        lengths: list[int] = []
        for start in range(0, len(ids), step):
            end = min(start + self.CHUNK_TOKENS, len(ids))
            sub_ids = ids[start:end]
            if not sub_ids:
                break
            chunk_text = self.tokenizer.decode(sub_ids, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            lengths.append(len(sub_ids))
            if end == len(ids):
                break
        return chunks, lengths

    @staticmethod
    def _chunk_by_tokens_openai(text: str, chunk_size: int = 28000) -> tuple[list[str], list[int]]:
        """Chunk text for OpenAI embeddings (approximate by characters)."""
        if not text:
            return [""], [0]
        chunks: list[str] = []
        lengths: list[int] = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
            lengths.append(len(chunk))
        return chunks, lengths

    def _embed_texts_hf(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using SentenceTransformers."""
        vecs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i : i + BATCH_SIZE]
            emb = self.model.encode(
                batch,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=self.NORMALIZE,
                show_progress_bar=False,
            )
            vecs.append(emb)
        return np.vstack(vecs) if vecs else np.zeros((0,))

    def _embed_texts_openai(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using the OpenAI embeddings API (one request per text)."""
        vecs: list[list[float]] = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model, input=text)
            vecs.append(response.data[0].embedding)
        return np.vstack(vecs) if vecs else np.zeros((0,))

    def _chunk_by_tokens(self, text: str) -> tuple[list[str], list[int]]:
        """Dispatch chunking according to backend."""
        if self.mode == "huggingface":
            return self._chunk_by_tokens_hf(text)
        if self.mode == "openai":
            return self._chunk_by_tokens_openai(text)
        raise ValueError(f"Unknown model mode: {self.mode}")

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Dispatch embedding according to backend."""
        if self.mode == "huggingface":
            return self._embed_texts_hf(texts)
        if self.mode == "openai":
            return self._embed_texts_openai(texts)
        raise ValueError(f"Unknown model mode: {self.mode}")

    def embed_with_chunking(self, text: str) -> np.ndarray:
        """Chunk long text, embed each chunk, then pool across chunks.

        Pooling:
        - "mean": length-weighted average across chunk embeddings
        - "max" : elementwise max across chunk embeddings
        """
        chunks, lengths = self._chunk_by_tokens(text)
        embs = self._embed_texts(chunks)  # (num_chunks, dim) or (dim,)

        if embs.ndim == 1:
            pooled = embs.astype(np.float32)
        else:
            if self.pooling == "mean":
                weights = np.array(lengths, dtype=np.float32)
                weights = np.maximum(weights, 1.0)
                weights = weights / weights.sum()
                pooled = (embs * weights[:, None]).sum(axis=0)
            elif self.pooling == "max":
                pooled = np.max(embs, axis=0)
            else:
                raise ValueError(f"Unknown pooling={self.pooling}")

        if self.NORMALIZE:
            norm = np.linalg.norm(pooled) + 1e-12
            pooled = pooled / norm

        return pooled.astype(np.float32)


def _coerce_list(x: object) -> List[str]:
    """Parse a value into a list of strings.

    Accepts:
    - list input
    - stringified Python literals (via `ast.literal_eval`)
    - JSON lists
    - delimiter-separated strings (| ; ,)
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [s.strip() for s in map(str, x) if str(s).strip()]

    s = str(x).strip()
    if not s:
        return []

    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(v).strip().strip('"').strip("'") for v in val if str(v).strip().strip('"').strip("'")]
    except Exception:
        pass

    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    for delim in ["|", ";", ","]:
        if delim in s:
            parts = [p.strip().strip('"').strip("'") for p in s.split(delim)]
            return [p for p in parts if p]
    return [s.strip().strip('"').strip("'")]


def _invert_team_mapping(short_to_full: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert {short -> full} mapping into {full -> [shorts]}."""
    inv: Dict[str, List[str]] = {}
    for short, full in (short_to_full or {}).items():
        if not full:
            continue
        inv.setdefault(str(full).strip(), []).append(str(short).strip())
    return inv


@dataclass
class LoLMatchMetadata:
    """Minimal match metadata used for entity masking in commentary text."""

    team1_name: str
    team2_name: str
    team1_players: List[str]
    team2_players: List[str]
    team1_champions: List[str]
    team2_champions: List[str]

    @classmethod
    def from_row(cls, row: pd.Series) -> "LoLMatchMetadata":
        """Create metadata from a dataset row containing team/player/champion columns."""
        return cls(
            str(row.get("team1_name", "")).strip(),
            str(row.get("team2_name", "")).strip(),
            _coerce_list(row.get("team1_players", [])),
            _coerce_list(row.get("team2_players", [])),
            _coerce_list(row.get("team1_champions", [])),
            _coerce_list(row.get("team2_champions", [])),
        )

    def _team_map(self) -> Dict[str, str]:
        """Create a replacement map for team names and known abbreviations."""
        mapping: Dict[str, str] = {}
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

    def _player_map(self) -> Dict[str, str]:
        """Create a replacement map for player names."""
        mapping: Dict[str, str] = {}
        for i, p in enumerate(self.team1_players, start=1):
            mapping[p] = f"<T1_PLAYER_{i}>"
        for i, p in enumerate(self.team2_players, start=1):
            mapping[p] = f"<T2_PLAYER_{i}>"
        return mapping

    def _champion_map(self) -> Dict[str, str]:
        """Create a replacement map for champion names."""
        mapping: Dict[str, str] = {}
        for i, c in enumerate(self.team1_champions, start=1):
            mapping[c] = f"<T1_CHAMPION_{i}>"
        for i, c in enumerate(self.team2_champions, start=1):
            mapping[c] = f"<T2_CHAMPION_{i}>"
        return mapping


def add_embeddings_to_df(
    df: pd.DataFrame,
    ef: EmbeddingFactory,
    text_col: str = "text",
) -> pd.DataFrame:
    """Add masked text and pooled embeddings to a dataframe row-by-row."""
    out_embs: list[np.ndarray] = []
    out_masked: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunk+Embed per row"):
        text = str(row[text_col])
        masked = text
        if ef.mask_team or ef.mask_player or ef.mask_champion:
            masked = ef.mask_text(row, text)
        out_masked.append(masked)
        out_embs.append(ef.embed_with_chunking(masked))

    df_out = df.copy()
    df_out["masked_text"] = out_masked
    df_out["embedding"] = [v.astype(np.float32).tolist() for v in out_embs]
    return df_out


if __name__ == "__main__":
    chunking_config = ChunkingConfig(max_tokens=512, overlap=256)

    config = TextAndEmbeddingConfig(
        masking_types=(),
        asr="corrected",
        pooling="mean",  # we use this by default in the thesis and do not vary it.
        model=EmbeddingModelConfig(
            provider="hf",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            shortcut="minilm",
        ),
        chunking=chunking_config,
    )

    embedding_factory = EmbeddingFactory(config)

    df = pd.read_csv("dataset_filtered.csv", index_col=[0])
    df_with_vecs = add_embeddings_to_df(df, embedding_factory, text_col="text")

    df_with_vecs.to_parquet(f"{config.to_string()}.parquet")
    print(f"{config.to_string()}.parquet")
