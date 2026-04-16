"""Modash-to-pipeline flow: IG creators in → TikTok filter → enrich both → score.

Option B pipeline:
  1. Read Modash CSV export (IG handles + whatever metadata Modash provides)
  2. Try each IG handle as a TikTok handle via yt-dlp (FREE)
  3. Drop creators without TikTok
  4. Enrich surviving creators: TikTok via yt-dlp (FREE) + IG via RapidAPI (PAID)
  5. Score creators and output ranked CSV

This minimizes RapidAPI spend by only paying for creators we'll actually use.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.acquisition.tiktok_checker import batch_check_tiktok
from src.acquisition.tiktok_enricher import enrich_tiktok_profile
from src.acquisition.instagram_enricher import (
    InstagramProfileData,
    batch_enrich_instagram,
)
from src.analyzers.content_fit import (
    ContentFitResult,
    batch_evaluate_content_fit,
    DEFAULT_CAMPAIGN_BRIEF,
)
from src.config import Settings
from src.models_creator import CreatorProfile, PlatformMetrics
from src.scoring.creator_signals import score_creator
from src.scoring.affiliate_readiness import rank_creators

logger = logging.getLogger(__name__)


@dataclass
class ModashCreator:
    """A creator from the Modash CSV export."""
    ig_handle: str
    name: str = ""
    ig_followers: int | None = None
    ig_engagement_rate: float | None = None
    ig_avg_likes: float | None = None
    niche: str = ""
    recent_collabs: str = ""
    # Populated after TikTok check
    has_tiktok: bool = False
    tiktok_handle: str = ""  # Defaults to same as ig_handle


@dataclass
class ModashPipelineStats:
    """Stats for the Modash pipeline run."""
    total_input: int = 0
    tiktok_found: int = 0
    tiktok_not_found: int = 0
    ig_enriched: int = 0
    ig_enrich_failed: int = 0
    tiktok_enriched: int = 0
    tiktok_enrich_failed: int = 0
    content_fit_passed: int = 0
    content_fit_failed: int = 0
    content_fit_llm_reviewed: int = 0
    scored: int = 0
    output_written: int = 0


@dataclass
class ScoredCreator:
    """Final scored output for a creator with both platforms."""
    ig_handle: str
    tiktok_handle: str
    name: str = ""
    recent_collabs: str = ""

    # IG profile data
    ig_followers: int | None = None
    ig_following: int | None = None
    ig_post_count: int | None = None
    ig_engagement_rate: float | None = None
    ig_avg_likes: float | None = None
    ig_avg_comments: float | None = None
    ig_bio: str | None = None
    ig_verified: bool = False
    ig_is_business: bool = False

    # TikTok profile data
    tiktok_avg_views: float | None = None
    tiktok_avg_likes: float | None = None
    tiktok_avg_comments: float | None = None
    tiktok_avg_shares: float | None = None
    tiktok_avg_saves: float | None = None
    tiktok_engagement_rate: float | None = None
    tiktok_video_count: int | None = None

    # Content-fit
    content_fit_score: float | None = None
    content_fit_passed: bool = True
    content_fit_method: str = ""
    content_fit_reason: str = ""

    # Affiliate readiness (final composite)
    affiliate_score: float | None = None
    affiliate_rank: int | None = None
    recommendation: str = ""
    aff_tiktok_trust: float | None = None
    aff_tiktok_performance: float | None = None
    aff_tiktok_consistency: float | None = None
    aff_ig_engagement: float | None = None
    aff_authenticity: float | None = None

    # Creator scores (0.0 - 1.0)
    engagement_score: float | None = None
    follower_quality_score: float | None = None
    consistency_score: float | None = None
    authenticity_score: float | None = None
    creator_score: float | None = None
    creator_tier: str | None = None
    creator_flags: str = ""


def read_modash_csv(path: str | Path) -> list[ModashCreator]:
    """Read a Modash CSV export and extract IG handles.

    Handles Modash-specific formats:
      - Abbreviated followers: "42.9k", "1.2M", "34k"
      - Percentage engagement rates: "11.85%", "4%"
      - Handle format: "@username"

    Modash exports vary in format, so we try multiple column name patterns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modash CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        logger.warning("Modash CSV is empty")
        return []

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find the IG handle column — Modash uses various names
    ig_col = _find_column(df, [
        "handle", "username", "instagram_username", "ig_username",
        "instagram_handle", "ig_handle", "instagram", "ig",
        "profile_url", "url", "instagram_url",
    ])
    if ig_col is None:
        raise ValueError(
            f"Cannot find Instagram handle column in Modash CSV. "
            f"Found columns: {list(df.columns)}"
        )

    # Optional columns
    name_col = _find_column(df, ["name", "display_name", "full_name", "creator_name"])
    followers_col = _find_column(df, [
        "followers", "follower_count", "ig_followers",
        "instagram_followers", "audience_size",
    ])
    er_col = _find_column(df, [
        "engagement_rate", "er", "engagement", "avg_engagement_rate",
        "ig_engagement_rate",
    ])
    likes_col = _find_column(df, [
        "avg_likes", "average_likes", "ig_avg_likes", "likes",
    ])
    niche_col = _find_column(df, [
        "niche", "category", "topic", "content_category", "topics",
    ])
    collabs_col = _find_column(df, [
        "recent_collabs", "collabs", "collaborations", "brands",
        "brand_collabs", "partnerships",
    ])

    creators: list[ModashCreator] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        raw_handle = str(row[ig_col]).strip()
        if not raw_handle or raw_handle == "nan":
            continue

        # Normalize: strip @, extract from URL if needed
        handle = _normalize_handle(raw_handle)
        if not handle or handle in seen:
            continue
        seen.add(handle)

        creator = ModashCreator(
            ig_handle=handle,
            name=str(row.get(name_col, "")).strip() if name_col else "",
            ig_followers=_parse_abbreviated_number(row.get(followers_col)) if followers_col else None,
            ig_engagement_rate=_parse_percentage(row.get(er_col)) if er_col else None,
            ig_avg_likes=_parse_abbreviated_number_float(row.get(likes_col)) if likes_col else None,
            niche=str(row.get(niche_col, "")).strip() if niche_col else "",
            recent_collabs=str(row.get(collabs_col, "")).strip() if collabs_col else "",
        )
        creators.append(creator)

    with_followers = sum(1 for c in creators if c.ig_followers is not None)
    logger.info(
        f"Loaded {len(creators)} unique IG creators from Modash CSV "
        f"({with_followers} with follower data, {len(creators) - with_followers} without)"
    )
    return creators


async def run_modash_pipeline(
    input_path: str,
    output_path: str,
    settings: Settings,
    campaign_brief: str = "",
    tiktok_concurrency: int = 5,
    ig_concurrency: int = 3,
    ig_post_count: int = 12,
) -> ModashPipelineStats:
    """Run the full Modash Option B pipeline.

    1. Read Modash CSV
    2. Check TikTok existence (FREE)
    3. Drop creators without TikTok
    4. Enrich IG via RapidAPI (PAID — only for survivors)
    5. Enrich TikTok via yt-dlp (FREE)
    6. Score and output
    """
    stats = ModashPipelineStats()

    # --- Step 1: Read Modash CSV ---
    logger.info("=" * 60)
    logger.info("STEP 1: Reading Modash CSV")
    logger.info("=" * 60)
    creators = read_modash_csv(input_path)
    stats.total_input = len(creators)

    if not creators:
        logger.error("No creators found in Modash CSV")
        return stats

    # --- Step 2: Check TikTok existence (FREE) ---
    logger.info("=" * 60)
    logger.info(f"STEP 2: Checking TikTok for {len(creators)} IG handles (free via yt-dlp)")
    logger.info("=" * 60)

    ig_handles = [c.ig_handle for c in creators]
    tiktok_results = await batch_check_tiktok(
        ig_handles,
        concurrency=tiktok_concurrency,
    )

    # Update creators with TikTok status
    for creator in creators:
        exists = tiktok_results.get(creator.ig_handle, False)
        creator.has_tiktok = exists
        if exists:
            creator.tiktok_handle = creator.ig_handle  # Same handle on both platforms

    survivors = [c for c in creators if c.has_tiktok]
    removed = [c for c in creators if not c.has_tiktok]
    stats.tiktok_found = len(survivors)
    stats.tiktok_not_found = len(removed)

    logger.info(
        f"TikTok filter: {stats.tiktok_found} found, "
        f"{stats.tiktok_not_found} removed"
    )

    if not survivors:
        logger.warning("No creators have TikTok — nothing to enrich")
        return stats

    # --- Step 3: Enrich IG via RapidAPI (PAID — only for survivors) ---
    logger.info("=" * 60)
    logger.info(
        f"STEP 3: Enriching {len(survivors)} IG profiles via RapidAPI "
        f"(~{len(survivors) * 2} API calls)"
    )
    logger.info("=" * 60)

    ig_data: dict[str, InstagramProfileData | None] = {}
    if settings.rapidapi_key:
        ig_data = await batch_enrich_instagram(
            handles=[c.ig_handle for c in survivors],
            rapidapi_key=settings.rapidapi_key,
            concurrency=ig_concurrency,
            post_count=ig_post_count,
        )
        stats.ig_enriched = sum(1 for v in ig_data.values() if v is not None)
        stats.ig_enrich_failed = len(survivors) - stats.ig_enriched
    else:
        logger.warning(
            "No RapidAPI key set (AFF_RAPIDAPI_KEY). "
            "Skipping IG enrichment — will score from TikTok data only."
        )

    # --- Step 4: Enrich TikTok via yt-dlp (FREE) ---
    logger.info("=" * 60)
    logger.info(f"STEP 4: Enriching {len(survivors)} TikTok profiles via yt-dlp (free)")
    logger.info("=" * 60)

    tiktok_data: dict[str, dict | None] = {}
    sem = asyncio.Semaphore(tiktok_concurrency)
    completed = 0

    async def _enrich_tiktok(handle: str) -> None:
        nonlocal completed
        async with sem:
            result = await enrich_tiktok_profile(handle, video_count=10)
            tiktok_data[handle] = result
            completed += 1
            status = "✓" if result else "✗"
            logger.info(f"[{completed}/{len(survivors)}] TikTok @{handle}: {status}")

    await asyncio.gather(*[_enrich_tiktok(c.tiktok_handle) for c in survivors])

    stats.tiktok_enriched = sum(1 for v in tiktok_data.values() if v is not None)
    stats.tiktok_enrich_failed = len(survivors) - stats.tiktok_enriched

    # --- Step 5: Content-Fit Filter ---
    logger.info("=" * 60)
    logger.info("STEP 5: Content-fit filter (keyword scan + LLM for ambiguous)")
    logger.info("=" * 60)

    # Build content bundles for each creator
    creators_content: list[dict] = []
    for creator in survivors:
        tk = tiktok_data.get(creator.tiktok_handle)
        ig = ig_data.get(creator.ig_handle)

        tk_descriptions = tk.get("descriptions", []) if tk else []
        ig_captions = [p.caption for p in ig.recent_posts if p.caption] if ig else []
        bio = ig.bio if ig and ig.bio else ""

        creators_content.append({
            "handle": creator.ig_handle,
            "tiktok_descriptions": tk_descriptions,
            "ig_captions": ig_captions,
            "bio": bio,
        })

    content_fit_results = await batch_evaluate_content_fit(
        creators_content=creators_content,
        campaign_brief=campaign_brief,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
        concurrency=3,
    )

    stats.content_fit_passed = sum(1 for r in content_fit_results.values() if r.passed)
    stats.content_fit_failed = sum(1 for r in content_fit_results.values() if not r.passed)
    stats.content_fit_llm_reviewed = sum(
        1 for r in content_fit_results.values() if r.method == "llm"
    )

    # --- Step 6: Score and output ---
    logger.info("=" * 60)
    logger.info("STEP 6: Scoring creators")
    logger.info("=" * 60)

    scored: list[ScoredCreator] = []
    for creator in survivors:
        tk = tiktok_data.get(creator.tiktok_handle)
        ig = ig_data.get(creator.ig_handle)
        fit = content_fit_results.get(creator.ig_handle)

        result = _score_creator(creator, tk, ig, settings)
        if result is not None:
            # Attach content-fit data
            if fit:
                result.content_fit_score = fit.score
                result.content_fit_passed = fit.passed
                result.content_fit_method = fit.method
                result.content_fit_reason = fit.reason
            scored.append(result)

    stats.scored = len(scored)

    # --- Step 7: Affiliate Readiness Ranking (content-fit passes only) ---
    logger.info("=" * 60)
    logger.info("STEP 7: Affiliate readiness ranking (content-fit passes only)")
    logger.info("=" * 60)

    # Build data bundles for affiliate scoring — only for content-fit passes
    fit_passed = [s for s in scored if s.content_fit_passed]
    fit_failed = [s for s in scored if not s.content_fit_passed]

    affiliate_input: list[dict] = []
    for s in fit_passed:
        tk = tiktok_data.get(s.tiktok_handle)
        ig = ig_data.get(s.ig_handle)
        tk_descriptions = tk.get("descriptions", []) if tk else []
        ig_captions = [p.caption for p in ig.recent_posts if p.caption] if ig else []

        affiliate_input.append({
            "handle": s.ig_handle,
            "content_fit_score": s.content_fit_score or 0.0,
            "tiktok_data": tk,
            "ig_data": ig,
            "bio": ig.bio if ig and ig.bio else "",
            "content_texts": tk_descriptions + ig_captions,
        })

    affiliate_results = await rank_creators(
        creators_data=affiliate_input,
        campaign_brief=campaign_brief,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
        concurrency=5,
    )

    # Merge affiliate scores back into scored creators
    aff_by_handle = {r["handle"]: r for r in affiliate_results}
    for s in fit_passed:
        aff = aff_by_handle.get(s.ig_handle)
        if aff:
            s.affiliate_score = aff["affiliate_score"]
            s.affiliate_rank = aff["rank"]
            s.recommendation = aff.get("recommendation", "")
            s.aff_tiktok_trust = aff.get("tiktok_trust")
            s.aff_tiktok_performance = aff.get("tiktok_performance")
            s.aff_tiktok_consistency = aff.get("tiktok_consistency")
            s.aff_ig_engagement = aff.get("ig_engagement")
            s.aff_authenticity = aff.get("authenticity")

    # Sort: content-fit passes first (by affiliate_score), then failures
    fit_passed.sort(key=lambda s: s.affiliate_score or 0, reverse=True)
    scored = fit_passed + fit_failed

    # Write output CSV
    _write_output(scored, output_path)
    stats.output_written = len(scored)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("MODASH PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Input (Modash):         {stats.total_input} IG creators")
    logger.info(f"  TikTok found:           {stats.tiktok_found}")
    logger.info(f"  TikTok not found:       {stats.tiktok_not_found}")
    logger.info(f"  IG enriched:            {stats.ig_enriched}")
    logger.info(f"  IG enrich failed:       {stats.ig_enrich_failed}")
    logger.info(f"  TikTok enriched:        {stats.tiktok_enriched}")
    logger.info(f"  TikTok enrich failed:   {stats.tiktok_enrich_failed}")
    logger.info(f"  Content-fit passed:     {stats.content_fit_passed}")
    logger.info(f"  Content-fit failed:     {stats.content_fit_failed}")
    logger.info(f"  Content-fit LLM reviewed: {stats.content_fit_llm_reviewed}")
    logger.info(f"  Scored & output:        {stats.scored}")
    logger.info(f"  Output:                 {output_path}")

    return stats


def _score_creator(
    creator: ModashCreator,
    tk_data: dict | None,
    ig_data: InstagramProfileData | None,
    settings: Settings,
) -> ScoredCreator | None:
    """Build a CreatorProfile from enriched data and score it."""
    profile = CreatorProfile(primary_handle=creator.ig_handle)

    # Build TikTok metrics
    if tk_data and tk_data.get("videos"):
        followers = creator.ig_followers  # Use IG followers as proxy (yt-dlp doesn't give TT followers)
        engagement_rate = None
        if followers and followers > 0 and tk_data.get("avg_likes") and tk_data.get("avg_comments"):
            avg_interactions = (tk_data["avg_likes"] or 0) + (tk_data["avg_comments"] or 0)
            engagement_rate = avg_interactions / followers * 100

        profile.tiktok = PlatformMetrics(
            platform="tiktok",
            handle=creator.tiktok_handle,
            display_name=tk_data.get("display_name"),
            bio=" | ".join(tk_data.get("descriptions", [])[:3]),
            followers=followers,
            video_count=tk_data.get("video_count_scraped"),
            avg_views=tk_data.get("avg_views"),
            avg_likes=tk_data.get("avg_likes"),
            avg_comments=tk_data.get("avg_comments"),
            avg_shares=tk_data.get("avg_shares"),
            avg_saves=tk_data.get("avg_saves"),
            engagement_rate=engagement_rate,
            view_counts=tk_data.get("view_counts", []),
            like_counts=tk_data.get("like_counts", []),
            comment_counts=tk_data.get("comment_counts", []),
            share_counts=tk_data.get("share_counts", []),
            save_counts=tk_data.get("save_counts", []),
            video_timestamps=[
                v["timestamp"] for v in tk_data.get("videos", [])
                if v.get("timestamp") is not None
            ],
        )

    # Build Instagram metrics
    if ig_data:
        profile.instagram = PlatformMetrics(
            platform="instagram",
            handle=creator.ig_handle,
            display_name=ig_data.display_name,
            bio=ig_data.bio,
            followers=ig_data.followers,
            following=ig_data.following,
            verified=ig_data.verified,
            post_count=ig_data.post_count,
            engagement_rate=ig_data.engagement_rate,
            avg_likes=ig_data.avg_likes,
            avg_comments=ig_data.avg_comments,
            like_counts=[p.likes for p in ig_data.recent_posts if p.likes > 0],
            comment_counts=[p.comments for p in ig_data.recent_posts if p.comments > 0],
        )

    # Compute combined metrics
    profile.compute_combined_metrics()

    # Skip if we have no usable data at all
    if profile.tiktok is None and profile.instagram is None:
        logger.warning(f"@{creator.ig_handle}: no enrichment data — skipping")
        return None

    # Score
    creator_score = score_creator(
        profile,
        weight_engagement=settings.creator_weight_engagement,
        weight_follower_quality=settings.creator_weight_follower_quality,
        weight_consistency=settings.creator_weight_consistency,
        weight_authenticity=settings.creator_weight_authenticity,
    )

    return ScoredCreator(
        ig_handle=creator.ig_handle,
        tiktok_handle=creator.tiktok_handle,
        name=creator.name,
        recent_collabs=creator.recent_collabs,
        # IG data
        ig_followers=ig_data.followers if ig_data else creator.ig_followers,
        ig_following=ig_data.following if ig_data else None,
        ig_post_count=ig_data.post_count if ig_data else None,
        ig_engagement_rate=ig_data.engagement_rate if ig_data else creator.ig_engagement_rate,
        ig_avg_likes=ig_data.avg_likes if ig_data else creator.ig_avg_likes,
        ig_avg_comments=ig_data.avg_comments if ig_data else None,
        ig_bio=ig_data.bio if ig_data else None,
        ig_verified=ig_data.verified if ig_data else False,
        ig_is_business=ig_data.is_business if ig_data else False,
        # TikTok data
        tiktok_avg_views=tk_data.get("avg_views") if tk_data else None,
        tiktok_avg_likes=tk_data.get("avg_likes") if tk_data else None,
        tiktok_avg_comments=tk_data.get("avg_comments") if tk_data else None,
        tiktok_avg_shares=tk_data.get("avg_shares") if tk_data else None,
        tiktok_avg_saves=tk_data.get("avg_saves") if tk_data else None,
        tiktok_engagement_rate=profile.tiktok.engagement_rate if profile.tiktok else None,
        tiktok_video_count=tk_data.get("video_count_scraped") if tk_data else None,
        # Scores
        engagement_score=creator_score.engagement_score,
        follower_quality_score=creator_score.follower_quality_score,
        consistency_score=creator_score.consistency_score,
        authenticity_score=creator_score.authenticity_score,
        creator_score=creator_score.creator_score,
        creator_tier=creator_score.tier,
        creator_flags=", ".join(creator_score.flags) if creator_score.flags else "",
    )


def _write_output(scored: list[ScoredCreator], path: str | Path) -> None:
    """Write scored creators to output CSV, ranked by score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, s in enumerate(scored, 1):
        rows.append({
            "rank": i,
            "name": s.name,
            "ig_handle": f"@{s.ig_handle}",
            "tiktok_handle": f"@{s.tiktok_handle}",
            "creator_score": _round(s.creator_score),
            "creator_tier": s.creator_tier or "",
            "affiliate_score": s.affiliate_score,
            "affiliate_rank": s.affiliate_rank,
            "recommendation": (s.recommendation or "")[:300],
            "aff_tiktok_trust": _round(s.aff_tiktok_trust),
            "aff_tiktok_performance": _round(s.aff_tiktok_performance),
            "aff_tiktok_consistency": _round(s.aff_tiktok_consistency),
            "aff_ig_engagement": _round(s.aff_ig_engagement),
            "aff_authenticity": _round(s.aff_authenticity),
            "content_fit_passed": s.content_fit_passed,
            "content_fit_score": _round(s.content_fit_score),
            "content_fit_method": s.content_fit_method,
            "content_fit_reason": (s.content_fit_reason or "")[:200],
            "recent_collabs": s.recent_collabs,
            # IG
            "ig_followers": s.ig_followers,
            "ig_following": s.ig_following,
            "ig_post_count": s.ig_post_count,
            "ig_engagement_rate": _round(s.ig_engagement_rate),
            "ig_avg_likes": _round(s.ig_avg_likes),
            "ig_avg_comments": _round(s.ig_avg_comments),
            "ig_verified": s.ig_verified,
            "ig_is_business": s.ig_is_business,
            "ig_bio": (s.ig_bio or "")[:200],  # Truncate long bios
            # TikTok
            "tiktok_avg_views": _round(s.tiktok_avg_views),
            "tiktok_avg_likes": _round(s.tiktok_avg_likes),
            "tiktok_avg_comments": _round(s.tiktok_avg_comments),
            "tiktok_avg_shares": _round(s.tiktok_avg_shares),
            "tiktok_avg_saves": _round(s.tiktok_avg_saves),
            "tiktok_engagement_rate": _round(s.tiktok_engagement_rate),
            "tiktok_video_count": s.tiktok_video_count,
            # Score breakdown
            "engagement_score": _round(s.engagement_score),
            "follower_quality_score": _round(s.follower_quality_score),
            "consistency_score": _round(s.consistency_score),
            "authenticity_score": _round(s.authenticity_score),
            "creator_flags": s.creator_flags,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(rows)} scored creators to {path}")


# --- Utility ---

import re


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_handle(value: str) -> str | None:
    """Normalize an Instagram handle from various formats.

    Accepts: "@username", "username", "https://instagram.com/username/"
    Returns: bare handle without @ or None.
    """
    if not value or value == "nan":
        return None

    value = value.strip().rstrip("/")

    # Extract from URL
    ig_match = re.search(r"instagram\.com/([^/?#]+)", value)
    if ig_match:
        return ig_match.group(1).lstrip("@")

    # Strip @ prefix
    return value.lstrip("@") if value else None


def _parse_abbreviated_number(val) -> int | None:
    """Parse abbreviated numbers like '42.9k', '1.2M', '34k', '10500'.

    Handles Modash format where followers are displayed as abbreviated strings.
    Returns integer or None.
    """
    if val is None:
        return None

    text = str(val).strip().replace(",", "")
    if not text or text == "nan":
        return None

    # Already a plain number
    try:
        return int(float(text))
    except ValueError:
        pass

    # Abbreviated: 42.9k, 1.2M, 34k
    text_upper = text.upper()
    multiplier = 1
    if text_upper.endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text_upper.endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text_upper.endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]

    try:
        return int(float(text) * multiplier)
    except (ValueError, TypeError):
        return None


def _parse_abbreviated_number_float(val) -> float | None:
    """Like _parse_abbreviated_number but returns float."""
    result = _parse_abbreviated_number(val)
    return float(result) if result is not None else None


def _parse_percentage(val) -> float | None:
    """Parse percentage strings like '11.85%', '4%', '2.91'.

    Returns the numeric value (e.g., 11.85 for '11.85%').
    """
    if val is None:
        return None

    text = str(val).strip()
    if not text or text == "nan":
        return None

    # Strip % sign
    text = text.rstrip("%").strip()

    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _round(val: float | None, decimals: int = 3) -> float | None:
    if val is None:
        return None
    return round(val, decimals)
