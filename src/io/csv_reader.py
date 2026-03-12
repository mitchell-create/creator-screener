from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.models import AffiliateInput

logger = logging.getLogger(__name__)

# Map common column name variations to our standardized names
COLUMN_ALIASES = {
    "profile_url": ["profile_url", "url", "tiktok_url", "tiktok_link", "link", "profile_link"],
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
    for _, row in df.iterrows():
        url = str(row["profile_url"]).strip()
        if not url or url == "nan":
            continue

        affiliate = AffiliateInput(
            profile_url=url,
            engagement_rate=_safe_float(row.get("engagement_rate")),
            followers=_safe_int(row.get("followers")),
        )
        affiliates.append(affiliate)

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
