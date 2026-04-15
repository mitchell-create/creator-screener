"""TikTok profile enrichment via yt-dlp.

Extracts per-video metadata (views, likes, comments, shares, saves,
descriptions) from a TikTok profile without downloading videos.
Aggregates into profile-level metrics for creator scoring.

This replaces Firecrawl for TikTok since Firecrawl blocks tiktok.com.
"""
from __future__ import annotations

import asyncio
import logging
import random

import yt_dlp

logger = logging.getLogger(__name__)


async def enrich_tiktok_profile(
    handle: str,
    video_count: int = 10,
    delay_range: tuple[float, float] = (1.0, 3.0),
) -> dict | None:
    """Scrape TikTok profile metadata via yt-dlp (no video download).

    Args:
        handle: TikTok handle (with or without @).
        video_count: Number of recent videos to pull metadata for.
        delay_range: Random delay range to avoid rate limiting.

    Returns:
        Dict with profile-level and per-video metrics, or None on failure.
    """
    clean_handle = handle.lstrip("@")
    url = f"https://tiktok.com/@{clean_handle}"

    # Random delay to be respectful
    delay = random.uniform(*delay_range)
    await asyncio.sleep(delay)

    def _extract() -> dict | None:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
            "retries": 3,
            "socket_timeout": 30,
            "playlistend": video_count,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = ydl.extract_info(url, download=False)
                if result is None:
                    return None

                entries = result.get("entries", [])
                if not entries:
                    return None

                # Extract per-video stats
                videos = []
                for entry in entries:
                    if entry is None:
                        continue
                    videos.append({
                        "video_id": entry.get("id", ""),
                        "description": entry.get("description", ""),
                        "duration": entry.get("duration"),
                        "view_count": entry.get("view_count"),
                        "like_count": entry.get("like_count"),
                        "comment_count": entry.get("comment_count"),
                        "repost_count": entry.get("repost_count"),
                        "save_count": entry.get("save_count"),
                        "upload_date": entry.get("upload_date"),
                        "timestamp": entry.get("timestamp"),
                    })

                if not videos:
                    return None

                # Aggregate profile-level metrics
                display_name = entries[0].get("channel") or entries[0].get("uploader") or clean_handle
                view_counts = [v["view_count"] for v in videos if v["view_count"] is not None]
                like_counts = [v["like_count"] for v in videos if v["like_count"] is not None]
                comment_counts = [v["comment_count"] for v in videos if v["comment_count"] is not None]
                share_counts = [v["repost_count"] for v in videos if v["repost_count"] is not None]
                save_counts = [v["save_count"] for v in videos if v["save_count"] is not None]

                avg_views = sum(view_counts) / len(view_counts) if view_counts else None
                avg_likes = sum(like_counts) / len(like_counts) if like_counts else None
                avg_comments = sum(comment_counts) / len(comment_counts) if comment_counts else None
                avg_shares = sum(share_counts) / len(share_counts) if share_counts else None
                avg_saves = sum(save_counts) / len(save_counts) if save_counts else None

                # Collect all descriptions for content analysis
                descriptions = [v["description"] for v in videos if v["description"]]

                return {
                    "handle": clean_handle,
                    "display_name": display_name,
                    "video_count_scraped": len(videos),
                    "videos": videos,
                    "view_counts": view_counts,
                    "like_counts": like_counts,
                    "comment_counts": comment_counts,
                    "share_counts": share_counts,
                    "save_counts": save_counts,
                    "avg_views": avg_views,
                    "avg_likes": avg_likes,
                    "avg_comments": avg_comments,
                    "avg_shares": avg_shares,
                    "avg_saves": avg_saves,
                    "descriptions": descriptions,
                }

        except Exception as e:
            logger.warning(f"yt-dlp enrichment failed for @{clean_handle}: {e}")
            return None

    try:
        return await asyncio.wait_for(asyncio.to_thread(_extract), timeout=90)
    except asyncio.TimeoutError:
        logger.warning(f"yt-dlp enrichment timed out for @{clean_handle}")
        return None
