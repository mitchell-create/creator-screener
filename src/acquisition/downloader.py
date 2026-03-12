from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


def _build_yt_dlp_opts(output_dir: Path, max_resolution: int = 720) -> dict:
    """Build yt-dlp options for downloading TikTok videos."""
    return {
        "format": f"best[height<={max_resolution}]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 60,
        "extract_flat": False,
        "noplaylist": True,
    }


async def download_video(
    video_url: str,
    output_dir: Path,
    max_resolution: int = 720,
    timeout: int = 120,
) -> Path | None:
    """Download a single video. Returns path to downloaded file or None on failure."""
    opts = _build_yt_dlp_opts(output_dir, max_resolution)

    def _download() -> Path | None:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if info is None:
                    return None
                filename = ydl.prepare_filename(info)
                filepath = Path(filename)
                if filepath.exists():
                    logger.debug(f"Downloaded: {filepath.name} ({filepath.stat().st_size / 1024:.0f}KB)")
                    return filepath
                return None
        except Exception as e:
            logger.warning(f"Failed to download {video_url}: {e}")
            return None

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_download),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Download timed out for {video_url}")
        return None


async def download_videos_batch(
    video_urls: list[str],
    output_dir: Path,
    concurrency: int = 10,
    max_resolution: int = 720,
    timeout: int = 120,
) -> dict[str, Path | None]:
    """Download multiple videos with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, Path | None] = {}

    async def _download_one(url: str) -> None:
        async with semaphore:
            path = await download_video(url, output_dir, max_resolution, timeout)
            results[url] = path

    tasks = [_download_one(url) for url in video_urls]
    await asyncio.gather(*tasks, return_exceptions=True)

    downloaded = sum(1 for v in results.values() if v is not None)
    logger.info(f"Downloaded {downloaded}/{len(video_urls)} videos")
    return results
