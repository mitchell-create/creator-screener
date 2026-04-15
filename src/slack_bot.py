"""Slack bot integration for AffiliatePipeline.

Watches a Slack channel for CSV uploads, runs the pipeline,
and posts results back to the thread.

Features:
- Progress updates posted to Slack after each batch
- Partial results saved to disk after each batch (crash recovery)
- React with :no_entry: to gracefully stop and get results so far
- All results (passed AND failed) stored in Supabase for dashboard viewing
- Passed affiliates auto-categorized by content type via Gemini
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

from src.config import Settings
from src.io.csv_reader import read_input_csv
from src.io.csv_writer import write_output_csv, write_passed_only_csv
from src.io.supabase_store import SupabaseStore
from src.models import AffiliateResult, PipelineStats
from src.pipeline.orchestrator import PipelineOrchestrator

from dataclasses import dataclass

logger = logging.getLogger("affiliate_pipeline.slack")

# Job queue: CSV uploads are queued and processed one at a time
_job_queue: asyncio.Queue | None = None
_queue_worker_task: asyncio.Task | None = None

# Reference to the currently-running orchestrator so we can signal stop
_active_orchestrator: PipelineOrchestrator | None = None


@dataclass
class _PipelineJob:
    channel: str
    thread_ts: str
    file_info: dict
    settings: Settings
    client: AsyncWebClient


def _create_supabase_store(settings: Settings) -> SupabaseStore | None:
    """Create a SupabaseStore if credentials are configured."""
    if settings.supabase_url and settings.supabase_key:
        return SupabaseStore(settings.supabase_url, settings.supabase_key)
    logger.warning("Supabase not configured (AFF_SUPABASE_URL / AFF_SUPABASE_KEY) -- results will NOT be persisted")
    return None


def create_app(settings: Settings) -> AsyncApp:
    """Create and configure the Slack Bolt async app."""
    app = AsyncApp(token=settings.slack_bot_token)

    @app.event("message")
    async def handle_message(event: dict, client: AsyncWebClient) -> None:
        """Handle messages -- look for CSV file uploads in the target channel."""
        channel = event.get("channel", "")
        subtype = event.get("subtype", "")

        # Ignore bot messages, message edits, etc.
        if subtype in ("bot_message", "message_changed", "message_deleted"):
            return

        # Only respond in the configured channel
        if settings.slack_channel_id and channel != settings.slack_channel_id:
            return

        # Check for file attachments
        files = event.get("files", [])
        if not files:
            return

        # Find CSV files
        csv_file = None
        for f in files:
            name = f.get("name", "")
            filetype = f.get("filetype", "")
            if filetype == "csv" or name.lower().endswith(".csv"):
                csv_file = f
                break

        if csv_file is None:
            return  # Not a CSV upload -- ignore silently

        thread_ts = event.get("ts", "")

        # Add job to queue
        global _job_queue, _queue_worker_task
        if _job_queue is None:
            _job_queue = asyncio.Queue()

        job = _PipelineJob(channel, thread_ts, csv_file, settings, client)
        queue_size = _job_queue.qsize()

        if queue_size > 0:
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    f":clipboard: *Queued* — {queue_size} job{'s' if queue_size > 1 else ''} ahead of you.\n"
                    f"You'll be notified when processing starts."
                ),
            )

        await _job_queue.put(job)

        # Start the queue worker if not already running
        if _queue_worker_task is None or _queue_worker_task.done():
            _queue_worker_task = asyncio.create_task(_queue_worker())

    @app.event("reaction_added")
    async def handle_reaction(event: dict, client: AsyncWebClient) -> None:
        """Handle reactions -- :no_entry: stops the pipeline gracefully."""
        global _active_orchestrator

        reaction = event.get("reaction", "")
        if reaction not in ("no_entry", "octagonal_sign", "stop_sign", "hand"):
            return

        if _active_orchestrator is not None and not _active_orchestrator.stop_event.is_set():
            _active_orchestrator.request_stop()

            channel = event.get("item", {}).get("channel", "")
            ts = event.get("item", {}).get("ts", "")
            if channel:
                await client.chat_postMessage(
                    channel=channel,
                    thread_ts=ts,
                    text=(
                        ":octagonal_sign: *Stop requested!*\n"
                        "The pipeline will finish its current batch and then return partial results."
                    ),
                )

    # Catch-all for unhandled events to prevent warnings
    @app.event({"type": "message", "subtype": "file_share"})
    async def handle_file_share(event: dict) -> None:
        pass  # Handled by the main message handler

    return app


async def _queue_worker() -> None:
    """Process pipeline jobs from the queue one at a time."""
    global _job_queue
    if _job_queue is None:
        return

    while not _job_queue.empty():
        job = await _job_queue.get()
        try:
            await _run_pipeline_job(
                job.channel, job.thread_ts, job.file_info, job.settings, job.client,
            )
        except Exception as e:
            logger.error(f"Queue worker caught unhandled error: {e}", exc_info=True)
        finally:
            _job_queue.task_done()


async def _run_pipeline_job(
    channel: str,
    thread_ts: str,
    file_info: dict,
    settings: Settings,
    client: AsyncWebClient,
) -> None:
    """Download CSV from Slack, run pipeline, post results back."""
    global _active_orchestrator

    store = _create_supabase_store(settings)
    run_id: str | None = None

    csv_path = None
    output_dir = None
    heartbeat_task: asyncio.Task | None = None
    _pipeline_start_time: float | None = None

    try:
        # 1. Post "processing started" message
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=_format_started_message(file_info),
        )

        # 2. Download CSV from Slack
        csv_path = await _download_slack_file(client, file_info, settings)
        logger.info(f"Downloaded CSV to {csv_path}")

        # 3. Read and validate CSV
        affiliates, instagram_handles = read_input_csv(csv_path)
        if not affiliates:
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    ":warning: *No valid affiliates found*\n"
                    "The CSV must contain a `profile_url` column with TikTok profile URLs."
                ),
            )
            return

        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=(
                f":mag: Found *{len(affiliates)}* affiliates. Running analysis...\n"
                f"_React with :no_entry: to stop early and get partial results._"
            ),
        )

        # 3b. Dedup: check for previously-scored affiliates
        dedup_results: list[AffiliateResult] = []
        new_affiliates = affiliates

        if store and settings.dedup_enabled:
            try:
                profile_urls = [a.profile_url for a in affiliates]
                existing = await store.fetch_existing_results(profile_urls)
                if existing:
                    dedup_results = list(existing.values())
                    new_affiliates = [a for a in affiliates if a.profile_url not in existing]
                    await client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=(
                            f":fast_forward: *Dedup:* {len(dedup_results)} already scored "
                            f"in previous runs — skipping.\n"
                            f"Analyzing *{len(new_affiliates)}* new affiliates."
                        ),
                    )
            except Exception as e:
                logger.warning(f"Dedup check failed, processing all: {e}")

        # 4. Create Supabase run record
        if store:
            try:
                run_id = await store.create_run(
                    source="slack",
                    source_file=file_info.get("name", "unknown.csv"),
                    total_input=len(affiliates),
                )
            except Exception as e:
                logger.warning(f"Failed to create Supabase run: {e}")

        # 5. Set up output directory for incremental saves
        output_dir = Path(tempfile.mkdtemp(prefix="aff_output_"))
        output_path = output_dir / "pipeline_results.csv"
        passed_path = output_dir / "pipeline_results_passed.csv"

        # 6. Create progress callback
        _last_heartbeat_time = time.monotonic()
        _pipeline_start_time = time.monotonic()

        async def on_batch_complete(
            batch_idx: int,
            total_batches: int,
            results_so_far: list[AffiliateResult],
            stats: PipelineStats,
        ) -> None:
            """Post progress to Slack, save to disk, and sync to Supabase."""
            nonlocal _last_heartbeat_time

            # Save partial results to disk
            write_output_csv(results_so_far, output_path)
            write_passed_only_csv(results_so_far, passed_path)
            logger.info(
                f"Saved partial results ({len(results_so_far)} affiliates) to {output_path}"
            )

            # Save to Supabase incrementally
            if store and run_id:
                try:
                    await store.save_affiliates(run_id, results_so_far)
                except Exception as e:
                    logger.warning(f"Failed to save batch to Supabase: {e}")

            # Post progress update to Slack
            pass_rate = (
                f"{stats.passed / stats.total_processed * 100:.0f}%"
                if stats.total_processed > 0
                else "N/A"
            )
            elapsed = time.monotonic() - _pipeline_start_time
            elapsed_str = _format_duration(elapsed)

            issues = []
            if stats.timeouts > 0:
                issues.append(f":warning: {stats.timeouts} timed out")
            if stats.errors > stats.timeouts:
                issues.append(f":x: {stats.errors - stats.timeouts} other errors")
            issues_line = "\n" + " | ".join(issues) if issues else ""

            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    f":bar_chart: *Progress: Batch {batch_idx}/{total_batches}* ({elapsed_str} elapsed)\n"
                    f"Processed: {stats.total_processed}/{stats.total_input} | "
                    f"Passed: {stats.passed} ({pass_rate}) | "
                    f"Errors: {stats.errors}"
                    f"{issues_line}"
                ),
            )
            _last_heartbeat_time = time.monotonic()

        async def _heartbeat_loop(
            stats: PipelineStats,
            total_input: int,
        ) -> None:
            """Post a status update to Slack every 30 minutes while the pipeline runs."""
            nonlocal _last_heartbeat_time
            while True:
                await asyncio.sleep(60)  # Check every minute
                since_last = time.monotonic() - _last_heartbeat_time
                if since_last < 1800:  # 30 minutes
                    continue
                elapsed = time.monotonic() - _pipeline_start_time
                elapsed_str = _format_duration(elapsed)
                pass_rate = (
                    f"{stats.passed / stats.total_processed * 100:.0f}%"
                    if stats.total_processed > 0
                    else "N/A"
                )
                issues = []
                if stats.timeouts > 0:
                    issues.append(f":warning: {stats.timeouts} timed out")
                if stats.errors > stats.timeouts:
                    issues.append(f":x: {stats.errors - stats.timeouts} other errors")
                issues_line = "\n" + " | ".join(issues) if issues else ""

                try:
                    await client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=(
                            f":heartbeat: *Still running* ({elapsed_str} elapsed)\n"
                            f"Processed: {stats.total_processed}/{total_input} | "
                            f"Passed: {stats.passed} ({pass_rate}) | "
                            f"Errors: {stats.errors}"
                            f"{issues_line}"
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Heartbeat message failed: {e}")
                _last_heartbeat_time = time.monotonic()

        # 6b. Creator enrichment (Tier 0)
        if settings.enrichment_enabled and new_affiliates:
            from src.pipeline.enrichment import CreatorEnrichmentPipeline
            from src.models import AffiliateInput

            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f":mag_right: *Enriching {len(new_affiliates)} creators* (scraping TikTok profiles)...",
            )
            try:
                enrichment = CreatorEnrichmentPipeline(settings=settings)
                enriched = await enrichment.enrich_affiliates(
                    new_affiliates,
                    instagram_handles=instagram_handles if instagram_handles else None,
                )
                before_count = len(new_affiliates)
                new_affiliates = [
                    AffiliateInput(
                        profile_url=e.profile_url,
                        engagement_rate=e.engagement_rate,
                        followers=e.followers,
                    )
                    for e in enriched
                ]
                filtered = before_count - len(new_affiliates)
                if filtered > 0:
                    await client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=f":scissors: Pre-filtered {filtered} low-scoring creators. {len(new_affiliates)} proceeding to video analysis.",
                    )
            except Exception as e:
                logger.warning(f"Enrichment failed, continuing without: {e}")

        # 7. Run the pipeline with progress tracking
        heartbeat_task = None
        if new_affiliates:
            orchestrator = PipelineOrchestrator(settings)
            _active_orchestrator = orchestrator
            stats = PipelineStats(total_input=len(affiliates))
            heartbeat_task = asyncio.create_task(
                _heartbeat_loop(stats, len(affiliates))
            )
            try:
                new_results, stats = await orchestrator.run(
                    new_affiliates,
                    on_batch_complete=on_batch_complete,
                )
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
        else:
            new_results = []
            stats = PipelineStats(total_input=len(affiliates))

        # 7b. Merge dedup results into output
        results = dedup_results + new_results
        stats.skipped_dedup = len(dedup_results)
        stats.total_input = len(affiliates)

        # 8. Write final output CSVs
        write_output_csv(results, output_path)
        write_passed_only_csv(results, passed_path)

        # 9. Final save to Supabase + categorize passed affiliates
        # Only save newly-processed results (deduped ones already exist)
        if store and run_id:
            affiliate_ids: list[str] = []
            try:
                affiliate_ids = await store.save_affiliates(run_id, new_results)
            except Exception as e:
                logger.warning(f"Failed to save affiliates to Supabase: {e}")

            try:
                await store.complete_run(run_id, stats)
            except Exception as e:
                logger.warning(f"Failed to complete Supabase run: {e}")

            try:
                passed_count = sum(1 for r in new_results if r.passed)
                if passed_count > 0 and affiliate_ids:
                    await _categorize_passed_affiliates(
                        new_results, affiliate_ids, store, settings, client, channel, thread_ts,
                    )
            except Exception as e:
                logger.warning(f"Failed to categorize affiliates: {e}")

        # 10. Upload result files to Slack
        await _upload_results(client, channel, thread_ts, output_path, passed_path)

        # 11. Post summary
        db_note = ""
        if store and run_id:
            db_note = f"\n:card_file_box: Results saved to Supabase (run `{run_id[:8]}...`)"

        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=_format_summary_message(stats) + db_note,
        )

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"CSV validation error: {e}")
        if store and run_id:
            try:
                await store.fail_run(run_id, str(e))
            except Exception:
                pass
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=(
                f":warning: *Invalid CSV*\n"
                f"`{e}`\n\n"
                f"The CSV must contain a `profile_url` column.\n"
                f"Accepted names: `profile_url`, `url`, `tiktok_url`, `tiktok_link`, `link`, `profile_link`"
            ),
        )
    except Exception as e:
        logger.error(f"Pipeline job failed: {e}", exc_info=True)

        if store and run_id:
            try:
                await store.fail_run(run_id, str(e))
            except Exception:
                pass

        # If we have partial results, upload them before reporting the error
        if output_dir and output_dir.exists():
            output_path = output_dir / "pipeline_results.csv"
            passed_path = output_dir / "pipeline_results_passed.csv"
            if output_path.exists() and output_path.stat().st_size > 100:
                try:
                    await _upload_results(
                        client, channel, thread_ts, output_path, passed_path
                    )
                    await client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=":floppy_disk: *Partial results uploaded above* (pipeline crashed mid-run)",
                    )
                except Exception:
                    logger.warning("Failed to upload partial results after crash")

        elapsed = time.monotonic() - _pipeline_start_time if _pipeline_start_time else None
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=_format_error_message(e, elapsed),
        )
    finally:
        _active_orchestrator = None
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        if store:
            await store.close()
        # Cleanup temp files
        if csv_path and Path(csv_path).exists():
            Path(csv_path).unlink(missing_ok=True)
        if output_dir and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)


async def _categorize_passed_affiliates(
    results: list[AffiliateResult],
    affiliate_ids: list[str],
    store: SupabaseStore,
    settings: Settings,
    client: AsyncWebClient,
    channel: str,
    thread_ts: str,
) -> None:
    """Use Gemini to categorize passed affiliates by content type."""
    from src.analyzers.gemini import ContentCategorizer

    if not settings.openrouter_api_key:
        return

    categorizer = ContentCategorizer(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )

    # Build a url->id mapping from the results
    url_to_db_id: dict[str, str] = {}
    for r, db_id in zip(results, affiliate_ids):
        if r.passed:
            url_to_db_id[r.profile_url] = db_id

    if not url_to_db_id:
        return

    categorized = 0
    for r in results:
        if not r.passed:
            continue
        db_id = url_to_db_id.get(r.profile_url)
        if not db_id:
            continue

        # Use the existing video frames if available, otherwise skip
        # We'll use the video scores' gemini data as a proxy — if the affiliate
        # went through Gemini analysis, frames were extracted to a temp dir.
        # Since temp dirs are cleaned up per-affiliate, we re-use profile metadata.
        # For now, categorize based on a text-only prompt using profile URL + scores.
        try:
            cat_result = await _categorize_from_profile(
                categorizer, r, settings,
            )
            if cat_result:
                await store.update_categories(
                    db_id,
                    category=cat_result["category"],
                    tags=cat_result["tags"],
                    confidence=cat_result["confidence"],
                )
                categorized += 1
        except Exception as e:
            logger.warning(f"Failed to categorize {r.profile_url}: {e}")

    if categorized > 0:
        logger.info(f"Categorized {categorized} passed affiliates")


async def _categorize_from_profile(
    categorizer,
    result: AffiliateResult,
    settings: Settings,
) -> dict | None:
    """Categorize an affiliate using a text-based Gemini call (no frames needed)."""
    import re

    if result.profile_url.startswith("@"):
        handle = result.profile_url[1:]
    else:
        handle_match = re.search(r"tiktok\.com/@([^/?#]+)", result.profile_url)
        handle = handle_match.group(1) if handle_match else result.profile_url

    prompt = f"""You are a content categorization expert. Based on this TikTok creator's profile and scoring data, categorize their content.

Creator: @{handle}
Followers: {result.followers or 'unknown'}
Engagement rate: {result.engagement_rate or 'unknown'}
Overall quality score: {result.overall_score}
Audio quality: {result.avg_audio_score}
Video aesthetic: {result.avg_video_aesthetic}
Video technical: {result.avg_video_technical}
Scene cuts avg: {result.avg_scene_cuts}

Provide:
1. category: The single best primary category from this list: beauty, fitness, fashion, food, comedy, lifestyle, tech, gaming, education, music, dance, pets, travel, health, parenting, diy, sports, automotive, finance, other
2. tags: 2-5 specific content tags (e.g. "skincare", "tutorials", "product-reviews", "meal-prep")
3. confidence: How confident you are in this categorization (0.0-1.0). Since you only have metadata and no visual content, confidence should generally be lower (0.3-0.6).

Respond ONLY with a JSON object. No other text.
Example: {{"category": "beauty", "tags": ["skincare", "tutorials"], "confidence": 0.4}}"""

    payload = {
        "model": settings.openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as http_client:
            resp = await http_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        import json
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        return {
            "category": str(parsed.get("category", "other")).lower(),
            "tags": [str(t).lower() for t in parsed.get("tags", [])],
            "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.4)))),
        }
    except Exception as e:
        logger.warning(f"Text-based categorization failed for @{handle}: {e}")
        return None


async def _download_slack_file(
    client: AsyncWebClient,
    file_info: dict,
    settings: Settings,
) -> Path:
    """Download a file from Slack to a temp location."""
    url = file_info.get("url_private_download") or file_info.get("url_private")
    if not url:
        raise ValueError("No download URL found for the uploaded file")

    temp_dir = Path(settings.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = temp_dir / f"slack_input_{file_info['id']}.csv"

    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(
            url,
            headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)

    return csv_path


async def _upload_results(
    client: AsyncWebClient,
    channel: str,
    thread_ts: str,
    output_path: Path,
    passed_path: Path,
) -> None:
    """Upload result CSV files to the Slack thread."""
    # Upload full results
    await client.files_upload_v2(
        channel=channel,
        thread_ts=thread_ts,
        file=str(output_path),
        filename="pipeline_results.csv",
        title="Pipeline Results (All Affiliates)",
    )

    # Only upload passed file if it has results beyond the header
    if passed_path.exists() and passed_path.stat().st_size > 100:
        await client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            file=str(passed_path),
            filename="pipeline_results_passed.csv",
            title="Pipeline Results (Passed Only)",
        )


# --- Message formatting ---


def _format_started_message(file_info: dict) -> str:
    filename = file_info.get("name", "unknown.csv")
    size_bytes = file_info.get("size", 0)
    size_kb = size_bytes / 1024 if size_bytes else 0
    return (
        f":hourglass_flowing_sand: *Pipeline started*\n"
        f"Processing `{filename}` ({size_kb:.1f} KB)\n"
        f"This may take several minutes depending on the number of affiliates.\n"
        f"_React with :no_entry: on your upload to stop early and get partial results._"
    )


def _format_summary_message(stats: PipelineStats) -> str:
    pass_rate = (
        f"{stats.passed / stats.total_processed * 100:.0f}%"
        if stats.total_processed > 0
        else "N/A"
    )

    status_icon = ":octagonal_sign:" if stats.stopped_early else ":white_check_mark:"
    status_text = "Pipeline stopped early (partial results)" if stats.stopped_early else "Pipeline complete"

    dedup_line = f"*Skipped (previously scored):* {stats.skipped_dedup}\n" if stats.skipped_dedup > 0 else ""

    issues_line = ""
    if stats.timeouts > 0 or stats.errors > 0:
        parts = []
        if stats.timeouts > 0:
            parts.append(f":warning: *{stats.timeouts} affiliates timed out* (metadata/download too slow)")
        non_timeout_errors = stats.errors - stats.timeouts
        if non_timeout_errors > 0:
            parts.append(f":x: *{non_timeout_errors} other errors* (check logs for details)")
        issues_line = "\n".join(parts) + "\n\n"

    return (
        f"{status_icon} *{status_text}*\n\n"
        f"{issues_line}"
        f"*Input:* {stats.total_input} affiliates\n"
        f"{dedup_line}"
        f"*Processed:* {stats.total_processed}\n"
        f"*Passed:* {stats.passed} ({pass_rate})\n"
        f"*Rejected:*\n"
        f"  \u2022 Tier 1 (Audio): {stats.rejected_tier1}\n"
        f"  \u2022 Tier 2 (Video): {stats.rejected_tier2}\n"
        f"  \u2022 Tier 3 (Gemini): {stats.rejected_tier3}\n"
        f"*Errors:* {stats.errors}\n"
        f"*Gemini cost:* ${stats.gemini_cost_usd:.4f}\n"
        f"*Borderline \u2192 Gemini:* {stats.borderline_sent_to_gemini}"
    )


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_min = minutes % 60
    return f"{hours}h {remaining_min}m"


def _format_error_message(error: Exception, elapsed_seconds: float | None = None) -> str:
    error_str = str(error)[:500]
    elapsed_line = ""
    if elapsed_seconds is not None:
        elapsed_line = f"\nFailed after {_format_duration(elapsed_seconds)} of processing.\n"
    return (
        f":x: *Pipeline failed*\n\n"
        f"Error: `{error_str}`\n"
        f"{elapsed_line}"
        f"Check Railway logs for full details."
    )


async def start_slack_bot() -> None:
    """Entry point: create app and start Socket Mode handler."""
    settings = Settings()

    if not settings.slack_bot_token:
        raise ValueError("AFF_SLACK_BOT_TOKEN is required for Slack mode")
    if not settings.slack_app_token:
        raise ValueError("AFF_SLACK_APP_TOKEN is required for Slack mode (Socket Mode)")

    logger.info("Starting AffiliatePipeline Slack bot (Socket Mode)...")
    if settings.slack_channel_id:
        logger.info(f"Watching channel: {settings.slack_channel_id}")
    else:
        logger.warning("No AFF_SLACK_CHANNEL_ID set -- bot will respond in ALL channels")

    app = create_app(settings)
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    await handler.start_async()
