from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable

from src.acquisition.downloader import download_video
from src.acquisition.metadata import get_recent_video_urls
from src.acquisition.storage import TempVideoStorage
from src.analyzers.audio import AudioAnalyzer
from src.analyzers.frame_extractor import extract_key_frames
from src.analyzers.gemini import GeminiAnalyzer
from src.analyzers.video import VideoAnalyzer
from src.config import Settings
from src.models import (
    AffiliateInput,
    AffiliateResult,
    PipelineStats,
    VideoScore,
)
from src.pipeline.batch_processor import chunk_list
from src.scoring.aggregator import aggregate_scores
from src.scoring.thresholds import evaluate_affiliate

logger = logging.getLogger(__name__)

# Type alias for the progress callback:
#   (batch_idx, total_batches, results_so_far, stats) -> None
BatchProgressCallback = Callable[
    [int, int, list[AffiliateResult], PipelineStats],
    Awaitable[None],
]


class PipelineOrchestrator:
    """Controls the tiered filtering pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = TempVideoStorage(settings.temp_dir)
        self.audio_analyzer = AudioAnalyzer(sample_rate=settings.audio_sample_rate)
        self.video_analyzer = VideoAnalyzer()
        self.gemini_analyzer: GeminiAnalyzer | None = None
        # Set this event to signal the pipeline to stop after the current batch
        self.stop_event = asyncio.Event()

        if settings.tier3_enabled and settings.openrouter_api_key:
            self.gemini_analyzer = GeminiAnalyzer(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                weekly_budget_usd=settings.gemini_weekly_budget_usd,
            )

    async def process_affiliate(
        self,
        affiliate: AffiliateInput,
        stats: PipelineStats,
    ) -> AffiliateResult:
        """Process a single affiliate through the tiered pipeline."""
        s = self.settings
        affiliate_id = self.storage.get_affiliate_id(affiliate.profile_url)
        affiliate_dir = self.storage.create_affiliate_dir(affiliate_id)

        result = AffiliateResult(
            profile_url=affiliate.profile_url,
            engagement_rate=affiliate.engagement_rate,
            followers=affiliate.followers,
        )

        try:
            # --- Step 1: Get video URLs ---
            # Reconstruct full URL for yt-dlp if profile_url is just a handle
            fetch_url = affiliate.profile_url
            if fetch_url.startswith("@"):
                fetch_url = f"https://tiktok.com/{fetch_url}"
            logger.info(f"[{affiliate_id}] Fetching video URLs...")
            video_metas = await get_recent_video_urls(
                fetch_url,
                count=s.videos_per_profile,
            )

            if not video_metas:
                result.error = "No videos found"
                stats.errors += 1
                return result

            # --- Step 2: Download videos (concurrently) ---
            logger.info(f"[{affiliate_id}] Downloading {len(video_metas)} videos...")
            download_tasks = [
                download_video(
                    meta.url,
                    affiliate_dir,
                    max_resolution=s.download_max_resolution,
                    timeout=s.download_timeout_seconds,
                )
                for meta in video_metas
            ]
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
            downloaded: dict[str, Path] = {}
            for meta, path in zip(video_metas, download_results):
                if isinstance(path, Exception):
                    logger.warning(f"[{affiliate_id}] Download failed for {meta.video_id}: {path}")
                elif path:
                    downloaded[meta.video_id] = path

            result.videos_downloaded = len(downloaded)
            if not downloaded:
                result.error = "Failed to download any videos"
                stats.errors += 1
                return result

            # --- Step 3: Tier 1 - Audio Analysis ---
            logger.info(f"[{affiliate_id}] Tier 1: Audio analysis...")
            video_scores: list[VideoScore] = []
            tier1_failures = 0

            for video_id, video_path in downloaded.items():
                audio_result = self.audio_analyzer.analyze(
                    video_path,
                    speech_min_pct=s.vad_speech_min_pct,
                    dnsmos_ovrl_min=s.dnsmos_ovrl_min,
                )
                vs = VideoScore(
                    video_id=video_id,
                    audio=audio_result,
                )
                video_scores.append(vs)
                if not audio_result.passed:
                    tier1_failures += 1
                    vs.tier_rejected = 1

            # If majority of videos fail audio, reject the affiliate
            failure_rate = tier1_failures / len(video_scores)
            if failure_rate > s.tier_failure_rate:
                result.video_scores = video_scores
                result.videos_analyzed = len(video_scores)
                result.passed = False
                result.rejection_tier = 1
                result.rejection_reason = f"Audio: {tier1_failures}/{len(video_scores)} videos failed"
                stats.rejected_tier1 += 1
                logger.info(f"[{affiliate_id}] REJECTED (Tier 1): {result.rejection_reason}")
                return result

            # --- Step 4: Tier 2 - Video Quality ---
            logger.info(f"[{affiliate_id}] Tier 2: Video quality analysis...")
            tier2_failures = 0

            for vs in video_scores:
                if vs.tier_rejected:
                    continue  # Skip already-rejected videos

                video_path = downloaded.get(vs.video_id)
                if video_path is None:
                    continue

                vq_result = self.video_analyzer.analyze(
                    video_path,
                    aesthetic_min=s.dover_aesthetic_min,
                    technical_min=s.dover_technical_min,
                    overall_min=s.dover_overall_min,
                    min_cuts=s.min_scene_cuts,
                )
                vs.video_quality = vq_result
                if not vq_result.passed:
                    tier2_failures += 1
                    vs.tier_rejected = 2

            # If majority of remaining videos fail video quality, reject
            analyzed_tier2 = sum(1 for vs in video_scores if vs.tier_rejected is None or vs.tier_rejected == 2)
            if analyzed_tier2 > 0 and (tier2_failures / analyzed_tier2) > s.tier_failure_rate:
                result.video_scores = video_scores
                result.videos_analyzed = len(video_scores)
                result.passed = False
                result.rejection_tier = 2
                result.rejection_reason = f"Video quality: {tier2_failures}/{analyzed_tier2} videos failed"
                stats.rejected_tier2 += 1
                logger.info(f"[{affiliate_id}] REJECTED (Tier 2): {result.rejection_reason}")
                return result

            # --- Step 5: Aggregate Scores ---
            result.video_scores = video_scores
            result.videos_analyzed = len(video_scores)

            agg = aggregate_scores(
                video_scores,
                weight_audio=s.weight_audio,
                weight_video=s.weight_video,
                weight_edits=s.weight_edits,
                weight_engagement=s.weight_engagement,
                engagement_rate=affiliate.engagement_rate,
            )
            result.avg_audio_score = agg.avg_audio
            result.avg_video_aesthetic = agg.avg_aesthetic
            result.avg_video_technical = agg.avg_technical
            result.avg_scene_cuts = agg.avg_cuts
            result.overall_score = agg.overall

            # --- Step 6: Threshold Check ---
            threshold_result = evaluate_affiliate(agg, s)
            result.passed = threshold_result.passed
            result.is_borderline = threshold_result.is_borderline
            result.rejection_reason = threshold_result.rejection_reason

            if not threshold_result.passed:
                result.rejection_tier = 2
                stats.rejected_tier2 += 1
                logger.info(f"[{affiliate_id}] REJECTED (Tier 2 aggregate): {result.rejection_reason}")
                return result

            # --- Step 7: Tier 3 - Gemini (borderline only) ---
            if (
                threshold_result.is_borderline
                and self.gemini_analyzer is not None
                and s.tier3_enabled
            ):
                logger.info(f"[{affiliate_id}] Tier 3: Gemini analysis (borderline)...")
                stats.borderline_sent_to_gemini += 1

                # Pick the best-scoring video for Gemini analysis
                best_video = max(
                    ((vid, path) for vid, path in downloaded.items()),
                    key=lambda x: x[0],  # Fallback: just pick first
                )
                frames = extract_key_frames(
                    best_video[1],
                    num_frames=s.gemini_max_frames,
                    output_dir=affiliate_dir / "frames",
                )

                if frames:
                    gemini_result = await self.gemini_analyzer.analyze(frames)
                    if gemini_result:
                        # Attach to the video score
                        for vs in video_scores:
                            if vs.video_id == best_video[0]:
                                vs.gemini = gemini_result
                                break

                        result.gemini_avg_score = gemini_result.overall

                        if (
                            gemini_result.overall is not None
                            and gemini_result.overall < s.gemini_min_score
                        ):
                            result.passed = False
                            result.rejection_tier = 3
                            reason = (
                                f"Gemini score {gemini_result.overall:.1f} "
                                f"< {s.gemini_min_score}"
                            )
                            if gemini_result.reasoning:
                                reason += f" — {gemini_result.reasoning}"
                            result.rejection_reason = reason
                            stats.rejected_tier3 += 1
                            logger.info(
                                f"[{affiliate_id}] REJECTED (Tier 3): {result.rejection_reason}"
                            )
                            return result

            # --- Passed all tiers ---
            if result.passed:
                stats.passed += 1
                logger.info(
                    f"[{affiliate_id}] PASSED (score: {result.overall_score:.3f})"
                )

            return result

        except Exception as e:
            logger.error(f"[{affiliate_id}] Pipeline error: {e}", exc_info=True)
            result.error = str(e)[:200]
            stats.errors += 1
            return result

        finally:
            # Always clean up temp files
            self.storage.cleanup_affiliate(affiliate_id)

    async def _process_with_semaphore(
        self,
        sem: asyncio.Semaphore,
        affiliate: AffiliateInput,
        stats: PipelineStats,
    ) -> AffiliateResult:
        """Process a single affiliate while respecting the concurrency semaphore."""
        timeout = self.settings.per_affiliate_timeout
        async with sem:
            try:
                result = await asyncio.wait_for(
                    self.process_affiliate(affiliate, stats),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                affiliate_id = self.storage.get_affiliate_id(affiliate.profile_url)
                logger.error(f"[{affiliate_id}] Timed out after {timeout}s")
                result = AffiliateResult(
                    profile_url=affiliate.profile_url,
                    engagement_rate=affiliate.engagement_rate,
                    followers=affiliate.followers,
                    error=f"Timed out after {timeout}s",
                )
                stats.errors += 1
                stats.timeouts += 1
            stats.total_processed += 1
            return result

    def request_stop(self) -> None:
        """Signal the pipeline to stop after the current batch finishes."""
        logger.info("Stop requested — will finish current batch and return partial results")
        self.stop_event.set()

    async def run(
        self,
        affiliates: list[AffiliateInput],
        limit: int | None = None,
        on_batch_complete: BatchProgressCallback | None = None,
    ) -> tuple[list[AffiliateResult], PipelineStats]:
        """Run the full pipeline on a list of affiliates.

        Args:
            affiliates: List of affiliates to process.
            limit: Optional limit on number of affiliates to process.
            on_batch_complete: Optional async callback invoked after each batch
                with (batch_idx, total_batches, results_so_far, stats).

        Returns:
            Tuple of (results, stats). If stopped early via stop_event,
            stats.stopped_early will be True and only completed results
            are returned.
        """
        if limit:
            affiliates = affiliates[:limit]

        self.stop_event.clear()
        stats = PipelineStats(total_input=len(affiliates))
        concurrency = self.settings.max_concurrent_analysis
        sem = asyncio.Semaphore(concurrency)

        batches = chunk_list(affiliates, self.settings.batch_size)
        logger.info(
            f"Processing {len(affiliates)} affiliates in {len(batches)} batches "
            f"(batch_size={self.settings.batch_size}, concurrency={concurrency})"
        )

        results: list[AffiliateResult] = []

        for batch_idx, batch in enumerate(batches):
            # Check for stop signal before starting a new batch
            if self.stop_event.is_set():
                logger.info(
                    f"Stop signal received — stopping after batch {batch_idx}/{len(batches)}. "
                    f"Returning {len(results)} partial results."
                )
                stats.stopped_early = True
                break

            logger.info(
                f"--- Batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} affiliates) ---"
            )

            tasks = [
                self._process_with_semaphore(sem, affiliate, stats)
                for affiliate in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Affiliate failed with exception: {result}")
                    err_result = AffiliateResult(
                        profile_url=batch[i].profile_url,
                        engagement_rate=batch[i].engagement_rate,
                        followers=batch[i].followers,
                        error=str(result)[:200],
                    )
                    results.append(err_result)
                    stats.errors += 1
                else:
                    results.append(result)

            disk_usage = self.storage.get_disk_usage_mb()
            logger.info(
                f"Batch {batch_idx + 1} complete. "
                f"Temp disk usage: {disk_usage:.1f}MB. "
                f"Passed so far: {stats.passed}/{stats.total_processed}"
            )

            # Fire the progress callback after each batch
            if on_batch_complete:
                try:
                    await on_batch_complete(batch_idx + 1, len(batches), results, stats)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        if self.gemini_analyzer:
            stats.gemini_cost_usd = self.gemini_analyzer.budget.spent_usd

        return results, stats
