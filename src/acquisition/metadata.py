from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    url: str
    video_id: str
    duration: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    title: str | None = None


def _build_metadata_opts() -> dict:
    """Build yt-dlp options for metadata extraction (no download)."""
    return {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
        "retries": 3,
        "socket_timeout": 30,
    }


async def get_recent_video_urls(
    profile_url: str,
    count: int = 5,
    delay_range: tuple[float, float] = (1.0, 3.0),
) -> list[VideoMetadata]:
    """
    Fetch the most recent video URLs and metadata from a TikTok profile.

    Uses yt-dlp to extract video info without downloading.
    Adds a random delay to avoid rate limiting.
    """
    # Add random delay to be respectful
    delay = random.uniform(*delay_range)
    await asyncio.sleep(delay)

    def _extract() -> list[VideoMetadata]:
        opts = _build_metadata_opts()
        # Fetch extra videos to account for skipping positions 2 and 3
        # (likely pinned). We keep video 1 (best content ceiling check),
        # skip 2-3, then take the rest as typical output.
        opts["playlistend"] = count + 2

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(profile_url, download=False)
                if result is None:
                    return []

                entries = result.get("entries", [])
                if not entries:
                    # Single video page, not a profile
                    return [_info_to_metadata(result)] if result.get("id") else []

                all_videos = []
                for entry in entries:
                    if entry is None:
                        continue
                    meta = _info_to_metadata(entry)
                    if meta:
                        all_videos.append(meta)

                # Keep video 1, skip 2 and 3 (likely pinned), take the rest
                selected = []
                for i, video in enumerate(all_videos):
                    if i == 0:
                        selected.append(video)
                    elif i >= 3:
                        selected.append(video)
                    if len(selected) >= count:
                        break

                return selected

        except Exception as e:
            logger.warning(f"Failed to get metadata for {profile_url}: {e}")
            return []

    try:
        return await asyncio.wait_for(asyncio.to_thread(_extract), timeout=60)
    except asyncio.TimeoutError:
        logger.warning(f"Metadata extraction timed out for {profile_url}")
        return []


def _info_to_metadata(info: dict) -> VideoMetadata | None:
    """Convert yt-dlp info dict to VideoMetadata."""
    video_id = info.get("id")
    if not video_id:
        return None

    webpage_url = info.get("webpage_url") or info.get("url", "")

    return VideoMetadata(
        url=webpage_url,
        video_id=str(video_id),
        duration=info.get("duration"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        title=info.get("title"),
    )
