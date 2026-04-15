"""Creator profile models for multi-platform enrichment and scoring.

These models represent the enriched creator data gathered from TikTok and
Instagram, unified into a single CreatorProfile for scoring.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlatformMetrics(BaseModel):
    """Metrics from a single platform (TikTok or Instagram)."""

    platform: str  # "tiktok" or "instagram"
    handle: str
    display_name: str | None = None
    bio: str | None = None
    followers: int | None = None
    following: int | None = None
    verified: bool = False

    # Platform-specific
    total_likes: int | None = None  # TikTok total likes
    video_count: int | None = None  # TikTok
    post_count: int | None = None  # Instagram

    # Derived from scraped recent content
    avg_views: float | None = None  # TikTok average views per video
    avg_likes: float | None = None  # Average likes per post/video
    avg_comments: float | None = None  # Average comments per post/video
    engagement_rate: float | None = None  # (likes+comments)/followers * 100
    view_counts: list[int] = Field(default_factory=list)  # Raw view counts for consistency calc

    # TikTok-specific per-video data (for richer scoring)
    like_counts: list[int] = Field(default_factory=list)
    comment_counts: list[int] = Field(default_factory=list)
    share_counts: list[int] = Field(default_factory=list)  # reposts
    save_counts: list[int] = Field(default_factory=list)
    video_timestamps: list[int] = Field(default_factory=list)  # Unix timestamps for posting frequency
    avg_shares: float | None = None
    avg_saves: float | None = None


class CreatorProfile(BaseModel):
    """Unified creator profile combining data from all platforms.

    This is the input to the creator scoring pipeline. It aggregates
    metrics from TikTok and Instagram into a single view of the creator.
    """

    # Primary identifier (TikTok handle from the original pipeline)
    primary_handle: str

    # Platform-specific metrics
    tiktok: PlatformMetrics | None = None
    instagram: PlatformMetrics | None = None

    # Combined / best-available metrics
    total_followers: int = 0  # Sum across platforms
    primary_engagement_rate: float | None = None  # Best available ER
    bio_combined: str = ""  # Concatenated bios for embedding

    def compute_combined_metrics(self) -> None:
        """Aggregate metrics across platforms after enrichment."""
        self.total_followers = 0
        bios = []

        for platform in [self.tiktok, self.instagram]:
            if platform is None:
                continue
            self.total_followers += platform.followers or 0
            if platform.bio:
                bios.append(platform.bio)

        self.bio_combined = " | ".join(bios)

        # Use TikTok ER as primary if available, else Instagram
        if self.tiktok and self.tiktok.engagement_rate is not None:
            self.primary_engagement_rate = self.tiktok.engagement_rate
        elif self.instagram and self.instagram.engagement_rate is not None:
            self.primary_engagement_rate = self.instagram.engagement_rate


class CreatorScore(BaseModel):
    """Scoring breakdown for a creator, computed before video analysis.

    All sub-scores are normalized to 0.0 - 1.0.
    """

    # Sub-scores
    engagement_score: float | None = None  # Normalized engagement rate
    follower_quality_score: float | None = None  # Follower/following ratio signal
    consistency_score: float | None = None  # Variance in views across videos
    authenticity_score: float | None = None  # Bot/fake detection (1.0 = human)
    brand_fit_score: float | None = None  # Embedding similarity to campaign brief

    # Composite
    creator_score: float | None = None  # Weighted composite of above

    # Metadata
    tier: str | None = None  # "nano", "micro", "mid", "macro", "mega"
    flags: list[str] = Field(default_factory=list)  # Warning flags

    @staticmethod
    def classify_tier(followers: int | None) -> str:
        """Classify creator into standard influencer tiers."""
        if followers is None or followers == 0:
            return "unknown"
        if followers < 1_000:
            return "nano"
        if followers < 10_000:
            return "micro"
        if followers < 100_000:
            return "mid"
        if followers < 1_000_000:
            return "macro"
        return "mega"


class DiscoveryInput(BaseModel):
    """Input for creator discovery mode."""

    # Campaign description for AI keyword extraction + brand-fit matching
    campaign_brief: str = ""

    # Direct search parameters
    hashtags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # Filters
    platforms: list[str] = Field(default_factory=lambda: ["tiktok", "instagram"])
    min_followers: int = 1_000
    max_followers: int = 10_000_000
    min_engagement_rate: float = 0.0  # Percentage

    # Limits
    max_creators: int = 100  # Stop discovery after finding this many


class EnrichedAffiliateInput(BaseModel):
    """Extended AffiliateInput that includes enriched creator data.

    This is what gets passed to the video analysis pipeline after
    enrichment and pre-scoring.
    """

    profile_url: str
    engagement_rate: float | None = None
    followers: int | None = None

    # New: enriched data
    creator_profile: CreatorProfile | None = None
    creator_score: CreatorScore | None = None

    # Instagram handle (if provided)
    instagram_handle: str | None = None
