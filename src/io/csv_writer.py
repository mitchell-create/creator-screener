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
