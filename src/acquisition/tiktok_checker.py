"""Lightweight TikTok handle existence checker via yt-dlp.

Much faster than full enrichment — uses extract_flat to check if a profile
exists and has videos without downloading metadata for each video.
Used as the first filter in the Modash pipeline: if a creator doesn't have
a TikTok, we skip them entirely.
"""
from __future__ import annotations

import asyncio
import logging
import random

import yt_dlp

logger = logging.getLogger(__name__)


async def check_tiktok_exists(
    handle: str,
    delay_range: tuple[float, float] = (0.5, 1.5),
) -> bool:
    """Check if a TikTok handle exists and has at least one video.

    This is a lightweight probe — much faster than full enrichment.
    Uses extract_flat mode to avoid pulling per-video metadata.

    Args:
        handle: TikTok handle (with or without @).
        delay_range: Random delay to avoid rate limiting.

    Returns:
        True if the handle exists and has videos, False otherwise.
    """
    clean_handle = handle.lstrip("@")
    url = f"https://tiktok.com/@{clean_handle}"

    delay = random.uniform(*delay_range)
    await asyncio.sleep(delay)

    def _check() -> bool:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Just check existence, don't pull video details
            "skip_download": True,
            "retries": 2,
            "socket_timeout": 20,
            "playlistend": 1,  # Only need to find 1 video
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(url, download=False)
                if result is None:
                    return False
                entries = result.get("entries", [])
                return len(entries) > 0
        except Exception:
            return False

    try:
        return await asyncio.wait_for(asyncio.to_thread(_check), timeout=30)
    except asyncio.TimeoutError:
        logger.debug(f"TikTok check timed out for @{clean_handle}")
        return False


async def batch_check_tiktok(
    handles: list[str],
    concurrency: int = 5,
    on_progress: callable | None = None,
) -> dict[str, bool]:
    """Check multiple TikTok handles concurrently.

    Args:
        handles: List of handles to check (IG handles to try on TikTok).
        concurrency: Max concurrent yt-dlp lookups.
        on_progress: Optional callback(checked, total, handle, exists).

    Returns:
        Dict mapping handle -> exists (True/False).
    """
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, bool] = {}
    total = len(handles)
    checked = 0

    async def _check_one(handle: str) -> None:
        nonlocal checked
        async with sem:
            exists = await check_tiktok_exists(handle)
            results[handle] = exists
            checked += 1

            status = "✓ found" if exists else "✗ not found"
            logger.info(f"[{checked}/{total}] TikTok @{handle}: {status}")

            if on_progress:
                try:
                    await on_progress(checked, total, handle, exists)
                except Exception:
                    pass

    tasks = [_check_one(h) for h in handles]
    await asyncio.gather(*tasks)

    found = sum(1 for v in results.values() if v)
    logger.info(
        f"TikTok check complete: {found}/{total} handles found "
        f"({total - found} removed)"
    )
    return results
