from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.models import AffiliateResult

logger = logging.getLogger(__name__)


def write_output_csv(
    results: list[AffiliateResult],
    path: str | Path,
    include_failed: bool = True,
) -> Path:
    """Write results to output CSV. Returns the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        if not include_failed and not r.passed:
            continue

        row = {
            "profile_url": r.profile_url,
            "engagement_rate": r.engagement_rate,
            "followers": r.followers,

            # Creator-level scores (Tier 0 enrichment)
            "creator_score": _round(r.creator_score),
            "creator_tier": r.creator_tier or "",
            "creator_engagement_score": _round(r.creator_engagement_score),
            "creator_follower_quality": _round(r.creator_follower_quality),
            "creator_consistency": _round(r.creator_consistency),
            "creator_authenticity": _round(r.creator_authenticity),
            "creator_brand_fit": _round(r.creator_brand_fit),
            "creator_flags": r.creator_flags or "",

            # Scraped platform data
            "tiktok_followers_scraped": r.tiktok_followers_scraped,
            "tiktok_following": r.tiktok_following,
            "tiktok_total_likes": r.tiktok_total_likes,
            "tiktok_engagement_rate_scraped": _round(r.tiktok_engagement_rate_scraped),
            "instagram_handle": r.instagram_handle or "",
            "instagram_followers": r.instagram_followers,
            "instagram_engagement_rate": _round(r.instagram_engagement_rate),

            # Video quality scores
            "videos_analyzed": r.videos_analyzed,
            "avg_audio_score": _round(r.avg_audio_score),
            "avg_video_aesthetic": _round(r.avg_video_aesthetic),
            "avg_video_technical": _round(r.avg_video_technical),
            "avg_scene_cuts": _round(r.avg_scene_cuts),
            "gemini_avg_score": _round(r.gemini_avg_score),
            "overall_score": _round(r.overall_score),
            "passed": r.passed,
            "rejection_reason": r.rejection_reason or "",
            "rejection_tier": r.rejection_tier or "",
            "error": r.error or "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    passed_count = sum(1 for r in results if r.passed)
    logger.info(
        f"Wrote {len(rows)} results to {path} "
        f"({passed_count} passed, {len(results) - passed_count} failed)"
    )
    return path


def write_passed_only_csv(results: list[AffiliateResult], path: str | Path) -> Path:
    """Write only passing affiliates to a separate CSV."""
    return write_output_csv(results, path, include_failed=False)


def _round(val: float | None, decimals: int = 3) -> float | None:
    if val is None:
        return None
    return round(val, decimals)
