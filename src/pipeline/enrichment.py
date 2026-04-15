"""Creator enrichment pipeline.

Sits between input parsing and video analysis. Scrapes creator profiles
from TikTok (via yt-dlp) and Instagram (via Firecrawl) to gather
per-video metrics, then computes creator-level scores.

Pipeline position:
  CSV/Discovery → **Enrichment** → Video Analysis (Tier 1/2/3)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from src.acquisition.tiktok_enricher import enrich_tiktok_profile
from src.analyzers.brand_fit import BrandFitAnalyzer
from src.config import Settings
from src.models import AffiliateInput
from src.models_creator import (
    CreatorProfile,
    CreatorScore,
    EnrichedAffiliateInput,
    PlatformMetrics,
)
from src.scoring.creator_signals import score_creator

logger = logging.getLogger(__name__)

# Type alias for progress callback
EnrichmentProgressCallback = Callable[[int, int], Awaitable[None]]


class CreatorEnrichmentPipeline:
    """Enriches affiliate inputs with scraped profile data and creator scores."""

    def __init__(
        self,
        settings: Settings,
        brand_fit: BrandFitAnalyzer | None = None,
    ):
        self.settings = settings
        self.brand_fit = brand_fit

    async def enrich_affiliates(
        self,
        affiliates: list[AffiliateInput],
        instagram_handles: dict[str, str] | None = None,
        on_progress: EnrichmentProgressCallback | None = None,
    ) -> list[EnrichedAffiliateInput]:
        """Enrich a list of affiliates with scraped profile data and scores.

        Args:
            affiliates: Raw affiliate inputs from CSV.
            instagram_handles: Optional mapping of tiktok_handle -> ig_handle.
            on_progress: Optional callback(completed, total) for progress tracking.

        Returns:
            List of enriched affiliates with creator profiles and scores.
        """
        total = len(affiliates)
        logger.info(f"Enriching {total} affiliates with creator profile data...")

        enriched: list[EnrichedAffiliateInput] = []
        sem = asyncio.Semaphore(self.settings.enrichment_concurrency)

        async def _enrich_one(idx: int, affiliate: AffiliateInput) -> EnrichedAffiliateInput:
            async with sem:
                result = await self._enrich_single(
                    affiliate,
                    ig_handle=instagram_handles.get(affiliate.profile_url) if instagram_handles else None,
                )
                if on_progress:
                    try:
                        await on_progress(idx + 1, total)
                    except Exception:
                        pass
                return result

        tasks = [_enrich_one(i, aff) for i, aff in enumerate(affiliates)]
        enriched = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out creators below minimum creator score if threshold is set
        min_score = self.settings.creator_score_min
        if min_score > 0:
            before = len(enriched)
            enriched = [
                e for e in enriched
                if e.creator_score is None  # Keep if we couldn't score (will be scored by video)
                or e.creator_score.creator_score is None
                or e.creator_score.creator_score >= min_score
            ]
            filtered = before - len(enriched)
            if filtered > 0:
                logger.info(
                    f"Pre-filtered {filtered}/{before} creators below "
                    f"creator_score threshold {min_score}"
                )

        logger.info(f"Enrichment complete: {len(enriched)} creators ready for video analysis")
        return enriched

    async def _enrich_single(
        self,
        affiliate: AffiliateInput,
        ig_handle: str | None = None,
    ) -> EnrichedAffiliateInput:
        """Enrich a single affiliate with profile data from TikTok + Instagram."""
        handle = affiliate.profile_url.lstrip("@")
        profile = CreatorProfile(primary_handle=handle)

        # --- TikTok enrichment via yt-dlp ---
        try:
            tk_data = await enrich_tiktok_profile(
                handle,
                video_count=10,
            )
            if tk_data:
                # Compute engagement rate from per-video data + CSV follower count
                followers = affiliate.followers
                engagement_rate = None
                if followers and followers > 0 and tk_data.get("avg_likes") and tk_data.get("avg_comments"):
                    avg_interactions = (tk_data["avg_likes"] or 0) + (tk_data["avg_comments"] or 0)
                    engagement_rate = avg_interactions / followers * 100

                profile.tiktok = PlatformMetrics(
                    platform="tiktok",
                    handle=tk_data["handle"],
                    display_name=tk_data.get("display_name"),
                    bio=" | ".join(tk_data.get("descriptions", [])[:3]),  # Use first 3 video descriptions as proxy
                    followers=followers,  # From CSV (yt-dlp doesn't provide profile-level followers)
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

                logger.info(
                    f"TikTok @{handle}: "
                    f"avg_views={tk_data.get('avg_views'):.0f}, "
                    f"avg_likes={tk_data.get('avg_likes'):.0f}, "
                    f"avg_comments={tk_data.get('avg_comments'):.0f}, "
                    f"ER={engagement_rate:.2f}%"
                    if engagement_rate else
                    f"TikTok @{handle}: "
                    f"avg_views={tk_data.get('avg_views')}, "
                    f"avg_likes={tk_data.get('avg_likes')}, "
                    f"avg_comments={tk_data.get('avg_comments')}"
                )
        except Exception as e:
            logger.warning(f"TikTok enrichment failed for @{handle}: {e}")

        # --- Instagram enrichment via Firecrawl (if available) ---
        if ig_handle and self.settings.firecrawl_api_key:
            try:
                from src.acquisition.firecrawl_client import FirecrawlScraper

                firecrawl = FirecrawlScraper(self.settings.firecrawl_api_key)
                ig_profile = await firecrawl.scrape_instagram_profile(ig_handle)
                await firecrawl.close()

                if ig_profile:
                    profile.instagram = PlatformMetrics(
                        platform="instagram",
                        handle=ig_profile.handle,
                        display_name=ig_profile.display_name,
                        bio=ig_profile.bio,
                        followers=ig_profile.followers,
                        following=ig_profile.following,
                        verified=ig_profile.verified,
                        post_count=ig_profile.post_count,
                        engagement_rate=ig_profile.avg_engagement_rate,
                    )
            except Exception as e:
                logger.warning(f"Instagram enrichment failed for @{ig_handle}: {e}")

        # Compute combined metrics
        profile.compute_combined_metrics()

        # Override CSV data with scraped data if we got better numbers
        enriched_followers = affiliate.followers
        enriched_er = affiliate.engagement_rate

        if profile.tiktok and profile.tiktok.engagement_rate is not None:
            enriched_er = profile.tiktok.engagement_rate

        # Score the creator
        creator_score_result = score_creator(
            profile,
            weight_engagement=self.settings.creator_weight_engagement,
            weight_follower_quality=self.settings.creator_weight_follower_quality,
            weight_consistency=self.settings.creator_weight_consistency,
            weight_authenticity=self.settings.creator_weight_authenticity,
        )

        # Brand-fit scoring (if campaign brief was set)
        if self.brand_fit:
            try:
                fit_score = await self.brand_fit.score_brand_fit(profile)
                if fit_score is not None:
                    creator_score_result.brand_fit_score = fit_score
            except Exception as e:
                logger.warning(f"Brand-fit scoring failed for @{handle}: {e}")

        return EnrichedAffiliateInput(
            profile_url=affiliate.profile_url,
            engagement_rate=enriched_er,
            followers=enriched_followers,
            creator_profile=profile,
            creator_score=creator_score_result,
            instagram_handle=ig_handle,
        )
