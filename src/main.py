from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.io.csv_reader import read_input_csv
from src.io.csv_writer import write_output_csv, write_passed_only_csv
from src.models import AffiliateInput, AffiliateResult
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger("affiliate_pipeline")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def _load_already_processed(output_path: str) -> set[str]:
    """Load profile_urls already in the output CSV (for resume support)."""
    path = Path(output_path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        if "profile_url" in df.columns:
            urls = set(df["profile_url"].dropna().str.strip())
            return urls
    except Exception:
        pass
    return set()


async def run_pipeline(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    campaign_brief: str = "",
) -> None:
    settings = Settings()

    # Read input
    logger.info(f"Reading input CSV: {input_path}")
    affiliates, instagram_handles = read_input_csv(input_path)
    if not affiliates:
        logger.error("No affiliates found in input CSV")
        sys.exit(1)

    logger.info(f"Loaded {len(affiliates)} affiliates")

    # Resume: skip affiliates already in the output file
    previous_results: list[AffiliateResult] = []
    if resume:
        already_done = _load_already_processed(output_path)
        if already_done:
            # Reload previous results so they get merged into final output
            prev_df = pd.read_csv(output_path)
            for _, row in prev_df.iterrows():
                previous_results.append(AffiliateResult(
                    profile_url=row.get("profile_url", ""),
                    engagement_rate=row.get("engagement_rate") if pd.notna(row.get("engagement_rate")) else None,
                    followers=row.get("followers") if pd.notna(row.get("followers")) else None,
                    videos_analyzed=int(row.get("videos_analyzed", 0)) if pd.notna(row.get("videos_analyzed")) else 0,
                    avg_audio_score=row.get("avg_audio_score") if pd.notna(row.get("avg_audio_score")) else None,
                    avg_video_aesthetic=row.get("avg_video_aesthetic") if pd.notna(row.get("avg_video_aesthetic")) else None,
                    avg_video_technical=row.get("avg_video_technical") if pd.notna(row.get("avg_video_technical")) else None,
                    avg_scene_cuts=row.get("avg_scene_cuts") if pd.notna(row.get("avg_scene_cuts")) else None,
                    gemini_avg_score=row.get("gemini_avg_score") if pd.notna(row.get("gemini_avg_score")) else None,
                    overall_score=row.get("overall_score") if pd.notna(row.get("overall_score")) else None,
                    passed=bool(row.get("passed")) if pd.notna(row.get("passed")) else False,
                    rejection_tier=int(row.get("rejection_tier")) if pd.notna(row.get("rejection_tier")) and str(row.get("rejection_tier")).strip() else None,
                    rejection_reason=row.get("rejection_reason") if pd.notna(row.get("rejection_reason")) else None,
                    error=row.get("error") if pd.notna(row.get("error")) else None,
                ))
            before = len(affiliates)
            affiliates = [a for a in affiliates if a.profile_url not in already_done]
            logger.info(f"Resume: skipping {before - len(affiliates)} already-processed affiliates, {len(affiliates)} remaining")
            if not affiliates:
                logger.info("All affiliates already processed — nothing to do")
                return

    if dry_run:
        logger.info("DRY RUN: Would process the following affiliates:")
        for a in affiliates[:20]:
            logger.info(f"  {a.profile_url} (ER: {a.engagement_rate}, Followers: {a.followers})")
        if len(affiliates) > 20:
            logger.info(f"  ... and {len(affiliates) - 20} more")
        return

    # --- Creator Enrichment (Tier 0) ---
    # Maps profile_url -> enrichment data, carried through to final results
    enrichment_data: dict[str, dict] = {}

    if settings.enrichment_enabled:
        from src.analyzers.brand_fit import BrandFitAnalyzer
        from src.pipeline.enrichment import CreatorEnrichmentPipeline

        logger.info("Running creator enrichment pipeline...")

        brand_fit: BrandFitAnalyzer | None = None
        if settings.brand_fit_enabled and campaign_brief:
            brand_fit = BrandFitAnalyzer(
                openrouter_api_key=settings.openrouter_api_key,
                embedding_model=settings.embedding_model,
                use_local=settings.use_local_embeddings,
            )
            await brand_fit.set_campaign_brief(campaign_brief)

        enrichment = CreatorEnrichmentPipeline(
            settings=settings,
            brand_fit=brand_fit,
        )
        enriched = await enrichment.enrich_affiliates(
            affiliates,
            instagram_handles=instagram_handles if instagram_handles else None,
        )

        # Stash enrichment data to merge into final results
        for e in enriched:
            data: dict = {}
            if e.creator_score:
                data["creator_score"] = e.creator_score.creator_score
                data["creator_engagement_score"] = e.creator_score.engagement_score
                data["creator_follower_quality"] = e.creator_score.follower_quality_score
                data["creator_consistency"] = e.creator_score.consistency_score
                data["creator_authenticity"] = e.creator_score.authenticity_score
                data["creator_brand_fit"] = e.creator_score.brand_fit_score
                data["creator_tier"] = e.creator_score.tier
                data["creator_flags"] = ", ".join(e.creator_score.flags) if e.creator_score.flags else None
            if e.creator_profile:
                tk = e.creator_profile.tiktok
                ig = e.creator_profile.instagram
                if tk:
                    data["tiktok_followers_scraped"] = tk.followers
                    data["tiktok_following"] = tk.following
                    data["tiktok_total_likes"] = tk.total_likes
                    data["tiktok_engagement_rate_scraped"] = tk.engagement_rate
                if ig:
                    data["instagram_handle"] = ig.handle
                    data["instagram_followers"] = ig.followers
                    data["instagram_engagement_rate"] = ig.engagement_rate
            enrichment_data[e.profile_url] = data

        # Update affiliates with enriched data (override CSV values with scraped data)
        affiliates = [
            AffiliateInput(
                profile_url=e.profile_url,
                engagement_rate=e.engagement_rate,
                followers=e.followers,
            )
            for e in enriched
        ]
        logger.info(f"Enrichment complete: {len(affiliates)} creators proceeding to video analysis")

    # Run pipeline with incremental saves after each batch
    output = Path(output_path)
    passed_path = output.with_stem(output.stem + "_passed")
    all_previous = previous_results  # keep reference for merging

    async def _save_progress(batch_idx, total_batches, results_so_far, batch_stats):
        merged = all_previous + results_so_far
        write_output_csv(merged, output)
        write_passed_only_csv(merged, passed_path)
        logger.info(f"Progress saved: {len(merged)} results written to {output}")

    orchestrator = PipelineOrchestrator(settings)
    results, stats = await orchestrator.run(
        affiliates, limit=limit, on_batch_complete=_save_progress,
    )

    # Merge enrichment data into results
    if enrichment_data:
        for r in results:
            edata = enrichment_data.get(r.profile_url)
            if edata:
                r.creator_score = edata.get("creator_score")
                r.creator_engagement_score = edata.get("creator_engagement_score")
                r.creator_follower_quality = edata.get("creator_follower_quality")
                r.creator_consistency = edata.get("creator_consistency")
                r.creator_authenticity = edata.get("creator_authenticity")
                r.creator_brand_fit = edata.get("creator_brand_fit")
                r.creator_tier = edata.get("creator_tier")
                r.creator_flags = edata.get("creator_flags")
                r.tiktok_followers_scraped = edata.get("tiktok_followers_scraped")
                r.tiktok_following = edata.get("tiktok_following")
                r.tiktok_total_likes = edata.get("tiktok_total_likes")
                r.tiktok_engagement_rate_scraped = edata.get("tiktok_engagement_rate_scraped")
                r.instagram_handle = edata.get("instagram_handle")
                r.instagram_followers = edata.get("instagram_followers")
                r.instagram_engagement_rate = edata.get("instagram_engagement_rate")

    # Merge with previous results if resuming
    if previous_results:
        results = previous_results + results

    # Final write
    write_output_csv(results, output)
    write_passed_only_csv(results, passed_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:          {stats.total_input}")
    if enrichment_data:
        logger.info(f"Enriched:             {len(enrichment_data)}")
        if stats.pre_filtered > 0:
            logger.info(f"Pre-filtered (Tier 0):{stats.pre_filtered}")
    logger.info(f"Total processed:      {stats.total_processed}")
    logger.info(f"Rejected (Tier 1):    {stats.rejected_tier1} (audio)")
    logger.info(f"Rejected (Tier 2):    {stats.rejected_tier2} (video)")
    logger.info(f"Rejected (Tier 3):    {stats.rejected_tier3} (gemini)")
    logger.info(f"Passed:               {stats.passed}")
    logger.info(f"Errors:               {stats.errors}")
    logger.info(f"Borderline -> Gemini: {stats.borderline_sent_to_gemini}")
    logger.info(f"Gemini cost:          ${stats.gemini_cost_usd:.4f}")
    logger.info(f"Output:               {output}")
    logger.info(f"Passed-only:          {passed_path}")


async def run_discovery(
    output_path: str,
    campaign_brief: str = "",
    hashtags: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Run creator discovery mode: LLM suggests creators, yt-dlp verifies them."""
    settings = Settings()

    if not settings.openrouter_api_key:
        logger.error("AFF_OPENROUTER_API_KEY is required for discovery mode")
        sys.exit(1)

    from src.discovery.discoverer import CreatorDiscoverer

    platforms = [p.strip() for p in settings.discovery_platforms.split(",")]

    discoverer = CreatorDiscoverer(
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
    )

    logger.info("Starting creator discovery...")
    results = await discoverer.discover(
        campaign_brief=campaign_brief,
        hashtags=hashtags,
        platforms=platforms,
        max_creators=limit or settings.discovery_max_creators,
    )

    # Write discovered handles to CSV (ready for pipeline input)
    output = Path(output_path)
    rows = []
    for creator in results:
        row = {
            "profile_url": f"@{creator['handle']}",
            "platform": creator.get("platform", "tiktok"),
            "verified": creator.get("verified", False),
            "display_name": creator.get("display_name", ""),
            "avg_views": creator.get("avg_views", ""),
            "avg_likes": creator.get("avg_likes", ""),
            "niche": creator.get("niche", ""),
            "reason": creator.get("reason", ""),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
        logger.info(f"Discovered {len(rows)} creators, written to {output}")
    else:
        logger.warning("No creators discovered")

    # Print summary
    verified_count = sum(1 for r in results if r.get("verified"))
    logger.info("=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Suggested:  {len(results)} creators")
    logger.info(f"  Verified:   {verified_count}")
    logger.info(f"  Unverified: {len(results) - verified_count}")
    logger.info(f"  Output:     {output}")
    logger.info(
        f"Run 'affiliate-pipeline --input {output}' to score these creators"
    )


async def _run_modash(
    input_path: str,
    output_path: str,
    settings: Settings,
) -> None:
    """Run the Modash pipeline: IG handles → TikTok filter → enrich → score."""
    from src.pipeline.modash_pipeline import run_modash_pipeline

    if not settings.rapidapi_key:
        logger.warning(
            "AFF_RAPIDAPI_KEY not set — IG enrichment will be skipped. "
            "Scoring will use TikTok data only."
        )

    await run_modash_pipeline(
        input_path=input_path,
        output_path=output_path,
        settings=settings,
        tiktok_concurrency=settings.modash_tiktok_concurrency,
        ig_concurrency=settings.modash_ig_concurrency,
        ig_post_count=settings.modash_ig_post_count,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="TikTok Affiliate Video Quality Analyzer"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "slack", "discover", "modash"],
        default=None,
        help=(
            "Run mode: 'cli' (default, batch CSV processing), "
            "'slack' (Slack bot), "
            "'discover' (find creators from hashtags/campaign brief), "
            "'modash' (IG creators from Modash → TikTok filter → score)"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to input CSV (default: from AFF_INPUT_CSV_PATH env var)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output CSV (default: from AFF_OUTPUT_CSV_PATH env var)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of affiliates to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run — skips affiliates already in the output CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse CSV and show what would be processed without running analysis",
    )
    parser.add_argument(
        "--campaign-brief",
        default="",
        help=(
            "Campaign description for brand-fit scoring and AI-powered discovery. "
            "E.g., 'Clean beauty brand targeting women 18-35 who love skincare routines'"
        ),
    )
    parser.add_argument(
        "--hashtags",
        nargs="+",
        default=None,
        help="Hashtags to search in discovery mode (e.g., --hashtags skincare beauty ugc)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Determine run mode
    mode = args.mode
    if mode is None:
        settings = Settings()
        mode = "cli"

    if mode == "modash":
        settings = Settings()
        input_path = args.input or settings.input_csv_path
        output_path = args.output or "./data/modash_scored.csv"
        asyncio.run(_run_modash(input_path, output_path, settings))
    elif mode == "slack":
        # Deferred import so slack-bolt is not required for CLI mode
        from src.slack_bot import start_slack_bot
        asyncio.run(start_slack_bot())
    elif mode == "discover":
        settings = Settings()
        output_path = args.output or "./data/discovered_creators.csv"
        asyncio.run(run_discovery(
            output_path=output_path,
            campaign_brief=args.campaign_brief,
            hashtags=args.hashtags,
            limit=args.limit,
        ))
    else:
        settings = Settings()
        input_path = args.input or settings.input_csv_path
        output_path = args.output or settings.output_csv_path
        asyncio.run(run_pipeline(
            input_path, output_path, args.limit, args.dry_run, args.resume,
            campaign_brief=args.campaign_brief,
        ))


if __name__ == "__main__":
    cli()
