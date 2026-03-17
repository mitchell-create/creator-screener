"""Supabase storage for affiliate pipeline results.

Uses the Supabase REST API (PostgREST) via httpx — no extra dependencies needed.
Stores every affiliate result (passed AND failed) with full scoring detail
for dashboard viewing and manual review.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from src.models import AffiliateResult, PipelineStats

logger = logging.getLogger(__name__)

# Timeout for Supabase API calls
_TIMEOUT = 30.0


def _extract_handle(profile_url: str) -> str | None:
    """Extract TikTok handle from a profile URL."""
    match = re.search(r"tiktok\.com/@([^/?#]+)", profile_url)
    return match.group(1) if match else None


class SupabaseStore:
    """Writes pipeline results to Supabase tables."""

    def __init__(self, supabase_url: str, supabase_key: str):
        self.base_url = supabase_url.rstrip("/")
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Dedup: fetch previously scored affiliates
    # ------------------------------------------------------------------

    async def fetch_existing_results(
        self,
        profile_urls: list[str],
    ) -> dict[str, AffiliateResult]:
        """Check for affiliates already scored in previous runs.

        Returns a dict of profile_url -> most recent AffiliateResult
        (only non-errored results). Used to skip re-analysis.
        """
        if not profile_urls:
            return {}

        client = await self._get_client()
        existing: dict[str, AffiliateResult] = {}

        # Batch URLs to avoid query string length limits
        batch_size = 50
        for i in range(0, len(profile_urls), batch_size):
            batch = profile_urls[i : i + batch_size]
            # PostgREST in-filter syntax
            quoted = ",".join(f'"{url}"' for url in batch)
            params = {
                "profile_url": f"in.({quoted})",
                "error": "is.null",
                "order": "created_at.desc",
                "select": (
                    "profile_url,engagement_rate,followers,"
                    "videos_downloaded,videos_analyzed,"
                    "avg_audio_score,avg_video_aesthetic,avg_video_technical,"
                    "avg_scene_cuts,gemini_avg_score,overall_score,"
                    "passed,is_borderline,rejection_tier,rejection_reason,created_at"
                ),
            }
            resp = await client.get(
                f"{self.base_url}/rest/v1/affiliates",
                headers=self.headers,
                params=params,
            )
            resp.raise_for_status()
            rows = resp.json()

            for row in rows:
                url = row["profile_url"]
                # Keep only the first (most recent) result per URL
                if url in existing:
                    continue
                existing[url] = AffiliateResult(
                    profile_url=url,
                    engagement_rate=row.get("engagement_rate"),
                    followers=row.get("followers"),
                    videos_downloaded=row.get("videos_downloaded", 0),
                    videos_analyzed=row.get("videos_analyzed", 0),
                    avg_audio_score=row.get("avg_audio_score"),
                    avg_video_aesthetic=row.get("avg_video_aesthetic"),
                    avg_video_technical=row.get("avg_video_technical"),
                    avg_scene_cuts=row.get("avg_scene_cuts"),
                    gemini_avg_score=row.get("gemini_avg_score"),
                    overall_score=row.get("overall_score"),
                    passed=row.get("passed", False),
                    is_borderline=row.get("is_borderline", False),
                    rejection_tier=row.get("rejection_tier"),
                    rejection_reason=row.get("rejection_reason"),
                )

        logger.info(f"Dedup: found {len(existing)}/{len(profile_urls)} already scored")
        return existing

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    async def create_run(
        self,
        source: str = "slack",
        source_file: str | None = None,
        total_input: int = 0,
    ) -> str:
        """Create a new pipeline run record. Returns the run ID."""
        client = await self._get_client()
        payload = {
            "source": source,
            "source_file": source_file,
            "total_input": total_input,
            "status": "running",
        }
        resp = await client.post(
            f"{self.base_url}/rest/v1/affiliate_runs",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        run_id = data[0]["id"] if isinstance(data, list) else data["id"]
        logger.info(f"Created Supabase run: {run_id}")
        return run_id

    async def complete_run(
        self,
        run_id: str,
        stats: PipelineStats,
    ) -> None:
        """Update a run record with final stats."""
        client = await self._get_client()
        status = "stopped" if stats.stopped_early else "completed"
        payload = {
            "total_processed": stats.total_processed,
            "passed": stats.passed,
            "rejected_tier1": stats.rejected_tier1,
            "rejected_tier2": stats.rejected_tier2,
            "rejected_tier3": stats.rejected_tier3,
            "errors": stats.errors,
            "stopped_early": stats.stopped_early,
            "gemini_cost_usd": stats.gemini_cost_usd,
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        resp = await client.patch(
            f"{self.base_url}/rest/v1/affiliate_runs?id=eq.{run_id}",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        logger.info(f"Completed Supabase run {run_id} with status={status}")

    async def fail_run(self, run_id: str, error: str) -> None:
        """Mark a run as failed."""
        client = await self._get_client()
        payload = {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        resp = await client.patch(
            f"{self.base_url}/rest/v1/affiliate_runs?id=eq.{run_id}",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Affiliate results
    # ------------------------------------------------------------------

    async def save_affiliates(
        self,
        run_id: str,
        results: list[AffiliateResult],
    ) -> list[str]:
        """Save affiliate results to Supabase. Returns list of affiliate IDs.

        Uses upsert (on run_id + profile_url) so this can be called
        incrementally after each batch without creating duplicates.
        """
        if not results:
            return []

        client = await self._get_client()

        # Build affiliate rows
        affiliate_rows = []
        for r in results:
            affiliate_rows.append({
                "run_id": run_id,
                "profile_url": r.profile_url,
                "tiktok_handle": _extract_handle(r.profile_url),
                "engagement_rate": r.engagement_rate,
                "followers": r.followers,
                "videos_downloaded": r.videos_downloaded,
                "videos_analyzed": r.videos_analyzed,
                "avg_audio_score": r.avg_audio_score,
                "avg_video_aesthetic": r.avg_video_aesthetic,
                "avg_video_technical": r.avg_video_technical,
                "avg_scene_cuts": r.avg_scene_cuts,
                "gemini_avg_score": r.gemini_avg_score,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "is_borderline": r.is_borderline,
                "rejection_tier": r.rejection_tier,
                "rejection_reason": r.rejection_reason,
                "error": r.error,
            })

        # Upsert affiliates (conflict on run_id + profile_url)
        upsert_headers = {
            **self.headers,
            "Prefer": "return=representation,resolution=merge-duplicates",
        }
        resp = await client.post(
            f"{self.base_url}/rest/v1/affiliates",
            headers=upsert_headers,
            json=affiliate_rows,
        )
        resp.raise_for_status()
        saved = resp.json()

        # Build a mapping of profile_url -> affiliate db id
        url_to_id: dict[str, str] = {}
        for row in saved:
            url_to_id[row["profile_url"]] = row["id"]

        # Save per-video scores for affiliates that have them
        video_rows = []
        for r in results:
            affiliate_db_id = url_to_id.get(r.profile_url)
            if not affiliate_db_id or not r.video_scores:
                continue

            for vs in r.video_scores:
                video_row: dict = {
                    "affiliate_id": affiliate_db_id,
                    "video_id": vs.video_id,
                    "tier_rejected": vs.tier_rejected,
                }

                # Tier 1: Audio
                if vs.audio:
                    video_row.update({
                        "audio_speech_pct": vs.audio.speech_pct,
                        "audio_dnsmos_ovrl": vs.audio.dnsmos_ovrl,
                        "audio_dnsmos_sig": vs.audio.dnsmos_sig,
                        "audio_dnsmos_bak": vs.audio.dnsmos_bak,
                        "audio_passed": vs.audio.passed,
                        "audio_rejection": vs.audio.rejection_reason,
                    })

                # Tier 2: Video quality
                if vs.video_quality:
                    video_row.update({
                        "video_aesthetic": vs.video_quality.dover_aesthetic,
                        "video_technical": vs.video_quality.dover_technical,
                        "scene_cuts": vs.video_quality.scene_cuts,
                        "video_passed": vs.video_quality.passed,
                        "video_rejection": vs.video_quality.rejection_reason,
                    })

                # Tier 3: Gemini
                if vs.gemini:
                    video_row.update({
                        "gemini_lighting": vs.gemini.lighting,
                        "gemini_composition": vs.gemini.composition,
                        "gemini_production": vs.gemini.production_value,
                        "gemini_brand_safety": vs.gemini.brand_safety,
                        "gemini_overall": vs.gemini.overall,
                        "gemini_reasoning": vs.gemini.reasoning,
                    })

                video_rows.append(video_row)

        if video_rows:
            # Delete existing videos for these affiliates first (re-run safe)
            affiliate_ids = list(url_to_id.values())
            for aff_id in affiliate_ids:
                await client.delete(
                    f"{self.base_url}/rest/v1/affiliate_videos?affiliate_id=eq.{aff_id}",
                    headers=self.headers,
                )

            # Insert fresh video rows
            resp = await client.post(
                f"{self.base_url}/rest/v1/affiliate_videos",
                headers=self.headers,
                json=video_rows,
            )
            resp.raise_for_status()

        logger.info(
            f"Saved {len(affiliate_rows)} affiliates and {len(video_rows)} video scores "
            f"to Supabase (run {run_id})"
        )
        return list(url_to_id.values())

    # ------------------------------------------------------------------
    # Content categorization (for passed affiliates)
    # ------------------------------------------------------------------

    async def update_categories(
        self,
        affiliate_id: str,
        category: str,
        tags: list[str],
        confidence: float,
    ) -> None:
        """Update an affiliate's content category after Gemini categorization."""
        client = await self._get_client()
        payload = {
            "content_category": category,
            "content_tags": tags,
            "category_confidence": confidence,
        }
        resp = await client.patch(
            f"{self.base_url}/rest/v1/affiliates?id=eq.{affiliate_id}",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
