"""Slack bot integration for AffiliatePipeline.

Watches a Slack channel for CSV uploads, runs the pipeline,
and posts results back to the thread.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

from src.config import Settings
from src.io.csv_reader import read_input_csv
from src.io.csv_writer import write_output_csv, write_passed_only_csv
from src.models import PipelineStats
from src.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger("affiliate_pipeline.slack")

# Module-level lock: only one pipeline job at a time
_pipeline_lock = asyncio.Lock()


def create_app(settings: Settings) -> AsyncApp:
    """Create and configure the Slack Bolt async app."""
    app = AsyncApp(token=settings.slack_bot_token)

    @app.event("message")
    async def handle_message(event: dict, client: AsyncWebClient) -> None:
        """Handle messages — look for CSV file uploads in the target channel."""
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
            return  # Not a CSV upload — ignore silently

        thread_ts = event.get("ts", "")

        # Check if pipeline is already running
        if _pipeline_lock.locked():
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    ":hourglass: *Pipeline is busy*\n"
                    "A job is currently running. Please wait for it to finish and try again."
                ),
            )
            return

        # Run pipeline in background task
        asyncio.create_task(
            _run_pipeline_job(channel, thread_ts, csv_file, settings, client)
        )

    # Catch-all for unhandled events to prevent warnings
    @app.event({"type": "message", "subtype": "file_share"})
    async def handle_file_share(event: dict) -> None:
        pass  # Handled by the main message handler

    return app


async def _run_pipeline_job(
    channel: str,
    thread_ts: str,
    file_info: dict,
    settings: Settings,
    client: AsyncWebClient,
) -> None:
    """Download CSV from Slack, run pipeline, post results back."""
    async with _pipeline_lock:
        csv_path = None
        output_dir = None

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
            affiliates = read_input_csv(csv_path)
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
                text=f":mag: Found *{len(affiliates)}* affiliates. Running analysis...",
            )

            # 4. Run the pipeline
            orchestrator = PipelineOrchestrator(settings)
            results, stats = await orchestrator.run(affiliates)

            # 5. Write output CSVs to temp directory
            output_dir = Path(tempfile.mkdtemp(prefix="aff_output_"))
            output_path = output_dir / "pipeline_results.csv"
            passed_path = output_dir / "pipeline_results_passed.csv"
            write_output_csv(results, output_path)
            write_passed_only_csv(results, passed_path)

            # 6. Upload result files to Slack
            await _upload_results(client, channel, thread_ts, output_path, passed_path)

            # 7. Post summary
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=_format_summary_message(stats),
            )

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"CSV validation error: {e}")
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
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=_format_error_message(e),
            )
        finally:
            # Cleanup temp files
            if csv_path and Path(csv_path).exists():
                Path(csv_path).unlink(missing_ok=True)
            if output_dir and output_dir.exists():
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)


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
        f"This may take several minutes depending on the number of affiliates."
    )


def _format_summary_message(stats: PipelineStats) -> str:
    pass_rate = (
        f"{stats.passed / stats.total_processed * 100:.0f}%"
        if stats.total_processed > 0
        else "N/A"
    )
    return (
        f":white_check_mark: *Pipeline complete*\n\n"
        f"*Input:* {stats.total_input} affiliates\n"
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


def _format_error_message(error: Exception) -> str:
    error_str = str(error)[:500]
    return (
        f":x: *Pipeline failed*\n\n"
        f"Error: `{error_str}`\n"
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
        logger.warning("No AFF_SLACK_CHANNEL_ID set — bot will respond in ALL channels")

    app = create_app(settings)
    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    await handler.start_async()
