from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import re

from src.models import AffiliateInput

logger = logging.getLogger(__name__)

# Regex to extract the TikTok handle from any tiktok.com URL
_TIKTOK_HANDLE_RE = re.compile(r"tiktok\.com/@([^/?#]+)")


def _normalize_to_handle(value: str) -> str:
    """Normalize any TikTok identifier to just the @handle.

    Accepts:
      - Full video URLs: 'https://tiktok.com/@user/video/123' -> '@user'
      - Profile URLs: 'https://tiktok.com/@user' -> '@user'
      - @handles: '@user' -> '@user'
      - Bare handles: 'user' -> '@user'
    """
    match = _TIKTOK_HANDLE_RE.search(value)
    if match:
        return f"@{match.group(1)}"
    # Already an @handle
    if value.startswith("@"):
        return value
    # Bare handle (no URL, no @) — prefix it
    if "/" not in value and "." not in value:
        return f"@{value}"
    return value

# Map common column name variations to our standardized names
COLUMN_ALIASES = {
    "profile_url": ["profile_url", "url", "tiktok_url", "tiktok_link", "link", "profile_link", "handle"],
    "engagement_rate": ["engagement_rate", "engagement", "er", "eng_rate"],
    "followers": ["followers", "follower_count", "follower", "subs", "subscribers"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace, lowercase, map aliases."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != standard_name:
                rename_map[alias] = standard_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def read_input_csv(path: str | Path) -> list[AffiliateInput]:
    """Read and validate the input CSV. Returns list of AffiliateInput models."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        logger.warning("Input CSV is empty")
        return []

    df = _normalize_columns(df)

    if "profile_url" not in df.columns:
        raise ValueError(
            f"Input CSV must have a profile URL column. "
            f"Found columns: {list(df.columns)}. "
            f"Accepted names: {COLUMN_ALIASES['profile_url']}"
        )

    affiliates = []
    seen_urls: set[str] = set()
    duplicates = 0
    normalized = 0

    for _, row in df.iterrows():
        url = str(row["profile_url"]).strip()
        if not url or url == "nan":
            continue

        # Normalize to @handle format (handles URLs, bare handles, etc.)
        original_url = url
        url = _normalize_to_handle(url)
        if url != original_url:
            normalized += 1

        # Deduplicate by normalized profile URL
        if url in seen_urls:
            duplicates += 1
            continue
        seen_urls.add(url)

        affiliate = AffiliateInput(
            profile_url=url,
            engagement_rate=_safe_float(row.get("engagement_rate")),
            followers=_safe_int(row.get("followers")),
        )
        affiliates.append(affiliate)

    if normalized > 0:
        logger.info(f"Normalized {normalized} video URLs to profile URLs")
    if duplicates > 0:
        logger.info(f"Skipped {duplicates} duplicate profile URLs")
    logger.info(f"Loaded {len(affiliates)} affiliates from {path}")
    return affiliates


def _safe_float(val) -> float | None:
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
