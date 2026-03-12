from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config import Settings
from src.io.csv_reader import read_input_csv
from src.io.csv_writer import write_output_csv, write_passed_only_csv
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


async def run_pipeline(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    settings = Settings()

    # Read input
    logger.info(f"Reading input CSV: {input_path}")
    affiliates = read_input_csv(input_path)
    if not affiliates:
        logger.error("No affiliates found in input CSV")
        sys.exit(1)

    logger.info(f"Loaded {len(affiliates)} affiliates")

    if dry_run:
        logger.info("DRY RUN: Would process the following affiliates:")
        for a in affiliates[:20]:
            logger.info(f"  {a.profile_url} (ER: {a.engagement_rate}, Followers: {a.followers})")
        if len(affiliates) > 20:
            logger.info(f"  ... and {len(affiliates) - 20} more")
        return

    # Run pipeline
    orchestrator = PipelineOrchestrator(settings)
    results, stats = await orchestrator.run(affiliates, limit=limit)

    # Write output
    output = Path(output_path)
    write_output_csv(results, output)

    passed_path = output.with_stem(output.stem + "_passed")
    write_passed_only_csv(results, passed_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total input:          {stats.total_input}")
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


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="TikTok Affiliate Video Quality Analyzer"
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
        "--dry-run",
        action="store_true",
        help="Parse CSV and show what would be processed without running analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    settings = Settings()
    input_path = args.input or settings.input_csv_path
    output_path = args.output or settings.output_csv_path

    asyncio.run(run_pipeline(input_path, output_path, args.limit, args.dry_run))


if __name__ == "__main__":
    cli()
