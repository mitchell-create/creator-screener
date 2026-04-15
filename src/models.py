from __future__ import annotations

from pydantic import BaseModel, Field


class AffiliateInput(BaseModel):
    profile_url: str
    engagement_rate: float | None = None
    followers: int | None = None


class AudioResult(BaseModel):
    speech_pct: float = 0.0
    dnsmos_ovrl: float | None = None
    dnsmos_sig: float | None = None
    dnsmos_bak: float | None = None
    passed: bool = False
    rejection_reason: str | None = None


class VideoQualityResult(BaseModel):
    dover_aesthetic: float | None = None
    dover_technical: float | None = None
    scene_cuts: int | None = None
    passed: bool = False
    rejection_reason: str | None = None


class GeminiResult(BaseModel):
    lighting: float | None = None
    composition: float | None = None
    production_value: float | None = None
    brand_safety: float | None = None
    overall: float | None = None
    reasoning: str | None = None


class VideoScore(BaseModel):
    video_url: str = ""
    video_id: str = ""
    # Tier 1
    audio: AudioResult | None = None
    # Tier 2
    video_quality: VideoQualityResult | None = None
    # Tier 3
    gemini: GeminiResult | None = None
    # Combined
    tier_rejected: int | None = None  # Which tier rejected (1, 2, 3, or None if passed)


class AffiliateResult(BaseModel):
    profile_url: str
    engagement_rate: float | None = None
    followers: int | None = None
    videos_analyzed: int = 0
    videos_downloaded: int = 0

    # Aggregated scores (averages across analyzed videos)
    avg_audio_score: float | None = None
    avg_video_aesthetic: float | None = None
    avg_video_technical: float | None = None
    avg_scene_cuts: float | None = None
    gemini_avg_score: float | None = None

    # Creator-level scores (from enrichment / Tier 0)
    creator_score: float | None = None
    creator_engagement_score: float | None = None
    creator_follower_quality: float | None = None
    creator_consistency: float | None = None
    creator_authenticity: float | None = None
    creator_brand_fit: float | None = None
    creator_tier: str | None = None  # nano, micro, mid, macro, mega
    creator_flags: str | None = None  # Comma-separated warning flags

    # Platform data (scraped)
    tiktok_followers_scraped: int | None = None
    tiktok_following: int | None = None
    tiktok_total_likes: int | None = None
    tiktok_engagement_rate_scraped: float | None = None
    instagram_handle: str | None = None
    instagram_followers: int | None = None
    instagram_engagement_rate: float | None = None

    # Final verdict
    overall_score: float | None = None
    passed: bool = False
    is_borderline: bool = False
    rejection_reason: str | None = None
    rejection_tier: int | None = None

    # Detailed per-video scores
    video_scores: list[VideoScore] = Field(default_factory=list)

    # Errors
    error: str | None = None


class PipelineStats(BaseModel):
    total_input: int = 0
    total_processed: int = 0
    rejected_tier1: int = 0
    rejected_tier2: int = 0
    rejected_tier3: int = 0
    passed: int = 0
    errors: int = 0
    borderline_sent_to_gemini: int = 0
    gemini_cost_usd: float = 0.0
    stopped_early: bool = False
    skipped_dedup: int = 0
    timeouts: int = 0

    # Enrichment stats
    enriched: int = 0
    enrichment_failed: int = 0
    pre_filtered: int = 0  # Creators filtered out by creator_score_min
