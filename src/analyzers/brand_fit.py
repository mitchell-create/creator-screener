"""Brand-fit matching via text embeddings.

Computes semantic similarity between a campaign brief and creator content
(bio + recent post descriptions) using OpenAI-compatible embeddings via
OpenRouter or a local sentence-transformers model.

Approach inspired by mangosqueezy's vector-similarity matching pattern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np

from src.models_creator import CreatorProfile

logger = logging.getLogger(__name__)


@dataclass
class BrandFitResult:
    """Result of brand-fit matching."""

    similarity: float  # 0.0 - 1.0 cosine similarity
    campaign_keywords: list[str]  # Keywords the LLM identified as relevant
    creator_topics: list[str]  # Topics extracted from creator bio/content


class BrandFitAnalyzer:
    """Scores how well a creator's content aligns with a campaign brief.

    Uses text embeddings to compute cosine similarity between:
      - Campaign brief / product description
      - Creator's bio + content descriptions

    Supports two backends:
      1. OpenRouter API (any OpenAI-compatible embedding model)
      2. Local sentence-transformers (free, no API calls)
    """

    def __init__(
        self,
        openrouter_api_key: str = "",
        embedding_model: str = "openai/text-embedding-3-small",
        use_local: bool = False,
    ):
        self.openrouter_api_key = openrouter_api_key
        self.embedding_model = embedding_model
        self.use_local = use_local
        self._local_model = None
        self._campaign_embedding: np.ndarray | None = None
        self._campaign_brief: str = ""

    async def set_campaign_brief(self, brief: str) -> None:
        """Pre-compute the campaign brief embedding for reuse across creators."""
        if not brief:
            return
        self._campaign_brief = brief
        self._campaign_embedding = await self._get_embedding(brief)
        if self._campaign_embedding is not None:
            logger.info("Campaign brief embedding computed and cached")

    async def score_brand_fit(self, profile: CreatorProfile) -> float | None:
        """Score how well a creator matches the campaign brief.

        Returns cosine similarity (0.0 - 1.0), or None if insufficient data.
        """
        if self._campaign_embedding is None:
            return None

        # Build creator text from available data
        creator_text = self._build_creator_text(profile)
        if not creator_text or len(creator_text.strip()) < 10:
            return None

        creator_embedding = await self._get_embedding(creator_text)
        if creator_embedding is None:
            return None

        # Cosine similarity
        similarity = self._cosine_similarity(self._campaign_embedding, creator_embedding)

        # Normalize: cosine similarity ranges from -1 to 1, but for text
        # embeddings it's typically 0 to 1. Clamp just in case.
        return max(0.0, min(1.0, similarity))

    def _build_creator_text(self, profile: CreatorProfile) -> str:
        """Build a text representation of the creator's content for embedding."""
        parts = []

        for platform in [profile.tiktok, profile.instagram]:
            if platform is None:
                continue
            if platform.bio:
                parts.append(platform.bio)

        # Use the combined bio if individual parts are empty
        if not parts and profile.bio_combined:
            parts.append(profile.bio_combined)

        return " ".join(parts)

    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding vector for text, using API or local model."""
        if self.use_local:
            return self._get_local_embedding(text)
        return await self._get_api_embedding(text)

    async def _get_api_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding via OpenRouter API (OpenAI-compatible endpoint)."""
        if not self.openrouter_api_key:
            logger.warning("No API key for embeddings — skipping brand-fit scoring")
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.embedding_model,
                        "input": text[:8000],  # Token limit safety
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = data["data"][0]["embedding"]
                return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Embedding API error: {e}")
            return None

    def _get_local_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding via local sentence-transformers model."""
        try:
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded local embedding model: all-MiniLM-L6-v2")

            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Local embedding error: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
