from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_key_frames(
    video_path: Path,
    num_frames: int = 4,
    output_dir: Path | None = None,
    max_width: int = 720,
    jpeg_quality: int = 80,
) -> list[Path]:
    """
    Extract evenly-spaced key frames from a video as JPEG files.

    Returns list of paths to frame images.
    """
    if output_dir is None:
        output_dir = video_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration
    duration = _get_duration(video_path)
    if duration is None or duration <= 0:
        logger.warning(f"Could not get duration for {video_path}")
        return []

    frames = []
    for i in range(num_frames):
        # Calculate timestamp for each frame (evenly spaced)
        timestamp = duration * (i + 1) / (num_frames + 1)
        frame_path = output_dir / f"{video_path.stem}_frame_{i:02d}.jpg"

        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", f"scale='min({max_width},iw)':-2",
                "-q:v", str(max(2, min(31, 32 - jpeg_quality // 3))),
                str(frame_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.returncode == 0 and frame_path.exists():
                frames.append(frame_path)
            else:
                logger.debug(f"Frame extraction failed at {timestamp:.1f}s")

        except Exception as e:
            logger.debug(f"Frame extraction error at {timestamp:.1f}s: {e}")

    logger.debug(f"Extracted {len(frames)}/{num_frames} frames from {video_path.name}")
    return frames


def _get_duration(video_path: Path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.debug(f"ffprobe failed: {e}")
    return None
