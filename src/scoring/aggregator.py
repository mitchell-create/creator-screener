from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models import VideoScore

logger = logging.getLogger(__name__)


@dataclass
class AggregateScore:
    avg_audio: float | None = None
    avg_aesthetic: float | None = None
    avg_technical: float | None = None
    avg_cuts: float | None = None
    gemini_avg: float | None = None
    overall: float | None = None


def aggregate_scores(
    video_scores: list[VideoScore],
    weight_audio: float = 0.35,
    weight_video: float = 0.35,
    weight_edits: float = 0.15,
    weight_engagement: float = 0.15,
    engagement_rate: float | None = None,
) -> AggregateScore:
    """Combine per-video scores into a single affiliate-level aggregate."""
    if not video_scores:
        return AggregateScore()

    # Collect valid scores per dimension
    audio_scores = []
    aesthetic_scores = []
    technical_scores = []
    cut_counts = []
    gemini_scores = []

    for vs in video_scores:
        if vs.audio and vs.audio.dnsmos_ovrl is not None:
            # Normalize DNSMOS 1-5 to 0-1
            audio_scores.append((vs.audio.dnsmos_ovrl - 1.0) / 4.0)

        if vs.video_quality:
            if vs.video_quality.dover_aesthetic is not None:
                aesthetic_scores.append(vs.video_quality.dover_aesthetic)
            if vs.video_quality.dover_technical is not None:
                technical_scores.append(vs.video_quality.dover_technical)
            if vs.video_quality.scene_cuts is not None:
                cut_counts.append(vs.video_quality.scene_cuts)

        if vs.gemini and vs.gemini.overall is not None:
            # Normalize 1-10 to 0-1
            gemini_scores.append((vs.gemini.overall - 1.0) / 9.0)

    avg_audio = _safe_mean(audio_scores)
    avg_aesthetic = _safe_mean(aesthetic_scores)
    avg_technical = _safe_mean(technical_scores)
    avg_cuts = _safe_mean(cut_counts)

    # Compute video quality as average of aesthetic and technical
    avg_video = None
    if avg_aesthetic is not None and avg_technical is not None:
        avg_video = (avg_aesthetic + avg_technical) / 2.0
    elif avg_aesthetic is not None:
        avg_video = avg_aesthetic
    elif avg_technical is not None:
        avg_video = avg_technical

    # Normalize edit count to 0-1 (cap at 10 cuts = 1.0)
    edit_score = None
    if avg_cuts is not None:
        edit_score = min(1.0, avg_cuts / 10.0)

    # Normalize engagement rate to 0-1 (cap at 10% = 1.0)
    engagement_score = None
    if engagement_rate is not None:
        engagement_score = min(1.0, engagement_rate / 10.0)

    gemini_avg = _safe_mean(gemini_scores)

    # Compute weighted overall score
    components = []
    total_weight = 0.0

    if avg_audio is not None:
        components.append(avg_audio * weight_audio)
        total_weight += weight_audio
    if avg_video is not None:
        components.append(avg_video * weight_video)
        total_weight += weight_video
    if edit_score is not None:
        components.append(edit_score * weight_edits)
        total_weight += weight_edits
    if engagement_score is not None:
        components.append(engagement_score * weight_engagement)
        total_weight += weight_engagement

    overall = sum(components) / total_weight if total_weight > 0 else None

    return AggregateScore(
        avg_audio=avg_audio,
        avg_aesthetic=avg_aesthetic,
        avg_technical=avg_technical,
        avg_cuts=avg_cuts,
        gemini_avg=gemini_avg,
        overall=overall,
    )


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
