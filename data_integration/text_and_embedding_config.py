from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

MaskingType = Literal["champion", "player", "team"]
ASRType = Literal["raw", "corrected"]
ProviderType = Literal["hf", "openai"]
PoolingType = Literal["mean", "max"]


@dataclass(frozen=True)
class ChunkingConfig:
    """Token-based chunking configuration (primarily for HF models)."""

    max_tokens: int
    overlap: int

    def to_string(self) -> str:
        """Compact identifier used in filenames, e.g. 't512-o256'."""
        return f"t{self.max_tokens}-o{self.overlap}"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Embedding model selection.

    Example:
      - provider="hf",     model_name="sentence-transformers/all-MiniLM-L6-v2", shortcut="minilm"
      - provider="openai", model_name="text-embedding-3-large",                shortcut="oai-e3l"
    """

    provider: ProviderType
    model_name: str
    shortcut: Optional[str] = None

    def id_string(self) -> str:
        """Compact identifier used in TextAndEmbeddingConfig.to_string().

        Uses `shortcut` if provided; otherwise sanitizes `model_name` for filenames.
        """
        core = self.shortcut or self.model_name.replace("/", "-").replace(" ", "")
        return f"{self.provider}-{core}"


@dataclass(frozen=True)
class TextAndEmbeddingConfig:
    """Text preprocessing + embedding configuration used to name datasets/outputs."""

    masking_types: Tuple[MaskingType, ...]
    asr: ASRType
    model: EmbeddingModelConfig
    chunking: Optional[ChunkingConfig] = None
    pooling: PoolingType = "mean"

    def to_string(self) -> str:
        """Create a compact, deterministic identifier for filenames / dataset IDs.

        Example:
          'm-team+player_asr-corr_mdl-hf-minilm_ck-t512-o256_pool-mean'
          'm-none_asr-raw_mdl-openai-oai-e3l_ck-none_pool-max'
        """
        mask_str = "+".join(sorted(self.masking_types)) if self.masking_types else "none"
        asr_str = "corr" if self.asr == "corrected" else "raw"
        model_str = self.model.id_string()
        ck_str = self.chunking.to_string() if self.chunking is not None else "none"
        return f"m-{mask_str}_asr-{asr_str}_mdl-{model_str}_ck-{ck_str}_pool-{self.pooling}"
