"""Platform-specific creator scoring signals for affiliate evaluation.

TikTok and Instagram have fundamentally different dynamics:

TikTok:
  - Algorithm-driven (FYP) → views are naturally inconsistent
  - Viral spikes are normal and expected, not a red flag
  - Saves and shares are stronger intent signals than likes
  - Posting frequency matters — active creators get algorithm boost
  - Engagement benchmarks are lower (1-3% is solid)
  - Like-to-view ratio is the real engagement signal (not likes/followers)

Instagram:
  - Follower-driven distribution → views/engagement are more predictable
  - Follower/following ratio is a strong authenticity signal
  - Engagement benchmarks are higher (3-6% is good)
  - Consistency matters more — erratic posting kills reach
  - Likes + comments per post relative to followers is the key metric

All scores are normalized to 0.0 - 1.0.
"""
from __future__ import annotations

import logging
import math
import statistics

from src.models_creator import CreatorProfile, CreatorScore, PlatformMetrics

logger = logging.getLogger(__name__)


def _follower_following_ratio(platform: PlatformMetrics) -> float | None:
    """Compute follower/following ratio for a platform."""
    if platform.followers and platform.following and platform.following > 0:
        return platform.followers / platform.following
    return None


def score_creator(
    profile: CreatorProfile,
    weight_engagement: float = 0.35,
    weight_follower_quality: float = 0.25,
    weight_consistency: float = 0.20,
    weight_authenticity: float = 0.20,
) -> CreatorScore:
    """Compute a composite creator score from enriched profile data.

    Automatically selects TikTok-specific or Instagram-specific scoring
    based on available data.
    """
    score = CreatorScore()
    flags: list[str] = []

    # Classify tier based on total followers
    score.tier = CreatorScore.classify_tier(profile.total_followers)

    # Route to platform-specific scoring
    if profile.tiktok and profile.tiktok.view_counts:
        # TikTok-specific scoring (primary path)
        score.engagement_score = _tiktok_engagement(profile.tiktok, flags)
        score.follower_quality_score = _tiktok_follower_quality(profile.tiktok, flags)
        score.consistency_score = _tiktok_consistency(profile.tiktok, flags)
        score.authenticity_score = _tiktok_authenticity(profile.tiktok, flags)
    elif profile.instagram:
        # Instagram-specific scoring
        score.engagement_score = _instagram_engagement(profile.instagram, flags)
        score.follower_quality_score = _instagram_follower_quality(profile.instagram, flags)
        score.consistency_score = _instagram_consistency(profile.instagram, flags)
        score.authenticity_score = _instagram_authenticity(profile.instagram, flags)
    else:
        # Fallback: use whatever CSV data we have
        score.engagement_score = _fallback_engagement(profile, flags)
        score.authenticity_score = _fallback_authenticity(profile, flags)

    # --- Weighted Composite ---
    components = []
    total_weight = 0.0

    if score.engagement_score is not None:
        components.append(score.engagement_score * weight_engagement)
        total_weight += weight_engagement
    if score.follower_quality_score is not None:
        components.append(score.follower_quality_score * weight_follower_quality)
        total_weight += weight_follower_quality
    if score.consistency_score is not None:
        components.append(score.consistency_score * weight_consistency)
        total_weight += weight_consistency
    if score.authenticity_score is not None:
        components.append(score.authenticity_score * weight_authenticity)
        total_weight += weight_authenticity

    if total_weight > 0:
        score.creator_score = sum(components) / total_weight
    else:
        flags.append("insufficient_data")

    score.flags = flags

    logger.info(
        f"Creator @{profile.primary_handle} score: "
        f"engagement={score.engagement_score}, "
        f"quality={score.follower_quality_score}, "
        f"consistency={score.consistency_score}, "
        f"authenticity={score.authenticity_score}, "
        f"composite={score.creator_score}, "
        f"tier={score.tier}, flags={flags}"
    )
    return score


# ======================================================================
# TikTok-Specific Scoring
# ======================================================================


def _tiktok_engagement(tk: PlatformMetrics, flags: list[str]) -> float | None:
    """TikTok engagement scoring.

    On TikTok, likes/followers engagement rate is misleading because views
    are algorithm-driven, not follower-driven. The real signals are:

    1. Like-to-view ratio (primary): What % of viewers liked?
       - >5% = excellent (score 0.8-1.0)
       - 3-5% = good (score 0.6-0.8)
       - 1-3% = average (score 0.3-0.6)
       - <1% = low quality or mass-push content (score 0.0-0.3)

    2. Save+share-to-view ratio (bonus): Indicates real intent
       - Saves = "I want to come back to this" (purchase intent)
       - Shares = "Others need to see this" (organic amplification)
       - >1% combined = strong signal → bonus
    """
    if not tk.view_counts or not tk.like_counts:
        return None

    # Compute per-video like-to-view ratios
    ltv_ratios = []
    for views, likes in zip(tk.view_counts, tk.like_counts):
        if views and views > 0 and likes is not None:
            ltv_ratios.append(likes / views * 100)

    if not ltv_ratios:
        return None

    # Use median to resist outliers (one viral video shouldn't skew it)
    median_ltv = statistics.median(ltv_ratios)

    # Logarithmic curve tuned for TikTok benchmarks:
    # 1% → 0.30, 3% → 0.55, 5% → 0.70, 8% → 0.83, 12%+ → 1.0
    base_score = math.log(1 + median_ltv) / math.log(1 + 12)
    base_score = min(1.0, max(0.0, base_score))

    # Save+share bonus (up to +0.15)
    bonus = 0.0
    if tk.save_counts and tk.share_counts and tk.view_counts:
        intent_ratios = []
        for views, saves, shares in zip(tk.view_counts, tk.save_counts, tk.share_counts):
            if views and views > 0:
                intent_ratios.append((saves + shares) / views * 100)
        if intent_ratios:
            median_intent = statistics.median(intent_ratios)
            # >1% intent ratio = meaningful, cap bonus at 2%
            bonus = min(0.15, median_intent * 0.075)

    final = min(1.0, base_score + bonus)

    if median_ltv < 0.5:
        flags.append("very_low_like_to_view")
    if median_ltv > 15:
        flags.append("unusually_high_like_to_view")

    return final


def _tiktok_follower_quality(tk: PlatformMetrics, flags: list[str]) -> float | None:
    """TikTok follower quality scoring.

    On TikTok, follower/following ratio is less meaningful because growth
    is algorithm-driven. Instead, we look at:

    1. Views-to-followers ratio: Are followers actually seeing content?
       - If avg views >> followers → algorithm is pushing content (good)
       - If avg views << followers → dead followers or shadowban (bad)

    2. Like-to-follower ratio per video: Are followers engaging?
    """
    if not tk.followers or tk.followers == 0:
        return None

    if not tk.view_counts:
        return None

    # Median views / followers ratio
    median_views = statistics.median(tk.view_counts) if tk.view_counts else 0
    view_follower_ratio = median_views / tk.followers

    # Scoring curve:
    # Ratio < 0.01 (1% of followers see content) → 0.1 (dead account)
    # Ratio 0.01-0.05 → 0.2-0.4 (low reach)
    # Ratio 0.05-0.20 → 0.4-0.7 (normal for large accounts)
    # Ratio 0.20-1.0 → 0.7-0.9 (good algorithm favor)
    # Ratio > 1.0 (more views than followers) → 0.9-1.0 (strong FYP presence)

    if view_follower_ratio < 0.01:
        flags.append("very_low_reach_ratio")
        return 0.1

    # Log curve: log(ratio*100 + 1) / log(200)
    # 1% → 0.17, 5% → 0.33, 20% → 0.56, 100% → 0.87, 500% → 1.0
    score = math.log(view_follower_ratio * 100 + 1) / math.log(200)
    return min(1.0, max(0.0, score))


def _tiktok_consistency(tk: PlatformMetrics, flags: list[str]) -> float | None:
    """TikTok consistency scoring — completely different from Instagram.

    On TikTok, high view variance is NORMAL because:
      - The algorithm decides distribution, not follower base
      - One video might get 10K views, the next 10M — that's fine
      - What matters for affiliates is the FLOOR, not the variance

    Scoring approach:
    1. Median views (the "reliable floor") — what you can expect
    2. Bottom quartile (P25) — worst-case scenario
    3. Floor-to-median ratio — is there a reliable base?

    We also factor in posting frequency as a positive signal.
    """
    if not tk.view_counts or len(tk.view_counts) < 3:
        return None

    sorted_views = sorted(tk.view_counts)
    median_views = statistics.median(sorted_views)
    p25_views = sorted_views[len(sorted_views) // 4] if len(sorted_views) >= 4 else sorted_views[0]

    if median_views == 0:
        return None

    # Floor reliability: P25 / median
    # If P25 is close to median → reliable floor (good)
    # If P25 is way below median → some videos tank (less predictable)
    floor_ratio = p25_views / median_views if median_views > 0 else 0

    # Scoring: floor_ratio of 0.5+ is great for TikTok (top quartile only 2x bottom)
    # floor_ratio of 0.1 means bottom videos get 10% of median (inconsistent but normal)
    # floor_ratio near 0 means some videos completely flop

    # Generous curve — TikTok variance is expected:
    # ratio 0.0 → 0.3 (still gets a base score — floor exists)
    # ratio 0.1 → 0.45
    # ratio 0.3 → 0.6
    # ratio 0.5 → 0.75
    # ratio 0.8+ → 0.9+
    score = 0.3 + 0.7 * (floor_ratio ** 0.5)  # Square root for gentle curve
    score = min(1.0, max(0.0, score))

    # Posting frequency bonus (if we have timestamps)
    if tk.video_timestamps and len(tk.video_timestamps) >= 2:
        timestamps = sorted(tk.video_timestamps)
        # Average days between posts
        gaps = [(timestamps[i + 1] - timestamps[i]) / 86400 for i in range(len(timestamps) - 1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 30

        # Posting at least every 3 days is good for TikTok
        if avg_gap <= 2:
            score = min(1.0, score + 0.08)  # Daily or near-daily poster
        elif avg_gap <= 5:
            score = min(1.0, score + 0.04)  # Regular poster
        elif avg_gap > 14:
            score = max(0.0, score - 0.05)  # Infrequent poster
            flags.append("low_posting_frequency")

    return score


def _tiktok_authenticity(tk: PlatformMetrics, flags: list[str]) -> float | None:
    """TikTok authenticity scoring.

    Signals that this is a real, active creator (not a bot or fake):
    1. Has meaningful content (video count)
    2. Has a display name and bio content
    3. Engagement looks organic (not all zeros, not impossibly uniform)
    4. Views have natural variance (bots often have flat metrics)
    """
    signals: list[float] = []

    # 1. Has bio / descriptions
    signals.append(1.0 if tk.bio and len(tk.bio) > 10 else 0.3)

    # 2. Has display name
    signals.append(1.0 if tk.display_name else 0.3)

    # 3. Meaningful video count
    count = tk.video_count or 0
    if count >= 20:
        signals.append(1.0)
    elif count >= 5:
        signals.append(0.7)
    elif count >= 1:
        signals.append(0.4)
    else:
        signals.append(0.0)
        flags.append("no_content")

    # 4. Organic engagement pattern
    # Real creators have varied like counts; bots often have flat or zero engagement
    if tk.like_counts and len(tk.like_counts) >= 3:
        non_zero = [lc for lc in tk.like_counts if lc > 0]
        if len(non_zero) >= 2:
            # Check that there's natural variance (not all identical)
            cv = statistics.stdev(non_zero) / statistics.mean(non_zero) if statistics.mean(non_zero) > 0 else 0
            if 0.1 < cv < 5.0:
                signals.append(1.0)  # Natural variance
            elif cv <= 0.1:
                signals.append(0.3)  # Suspiciously uniform
                flags.append("uniform_engagement_pattern")
            else:
                signals.append(0.6)  # Very high variance but still real
        elif len(non_zero) == 0:
            signals.append(0.1)
            flags.append("zero_engagement")
        else:
            signals.append(0.5)

    # 5. Comments exist (bots rarely generate real comments)
    if tk.comment_counts:
        has_comments = sum(1 for c in tk.comment_counts if c and c > 0)
        signals.append(min(1.0, has_comments / max(1, len(tk.comment_counts))))

    # 6. Saves exist (very hard to fake)
    if tk.save_counts:
        has_saves = sum(1 for s in tk.save_counts if s and s > 0)
        signals.append(min(1.0, has_saves / max(1, len(tk.save_counts))))

    if not signals:
        return None

    return sum(signals) / len(signals)


# ======================================================================
# Instagram-Specific Scoring
# ======================================================================


def _instagram_engagement(ig: PlatformMetrics, flags: list[str]) -> float | None:
    """Instagram engagement scoring.

    On Instagram, engagement is follower-driven. The standard metric is:
      ER = (likes + comments) / followers * 100

    Benchmarks:
      - >6% = excellent
      - 3-6% = good
      - 1-3% = average
      - <1% = low (possible dead followers or mass-follow growth)
      - >15% = suspicious (possible fake engagement)
    """
    er = ig.engagement_rate
    if er is None:
        return None

    if er > 15.0:
        flags.append("suspiciously_high_engagement")

    # Logarithmic curve tuned for Instagram benchmarks:
    # 1% → 0.30, 3% → 0.55, 6% → 0.75, 10% → 0.90, 15%+ → 1.0
    score = math.log(1 + er) / math.log(1 + 15)
    return min(1.0, max(0.0, score))


def _instagram_follower_quality(ig: PlatformMetrics, flags: list[str]) -> float | None:
    """Instagram follower quality — follower/following ratio matters here.

    On Instagram (unlike TikTok), the follower/following ratio is a strong
    signal of organic growth because distribution is follower-based.

    Scoring:
      - Ratio < 1.0 → 0.1 (following more than followers)
      - Ratio 1-2 → 0.3 (follow-for-follow territory)
      - Ratio 2-10 → 0.5-0.8 (normal growth)
      - Ratio 10-100 → 0.8-0.95 (strong organic audience)
      - Ratio 100+ → 0.95-1.0 (celebrity-level organic)
    """
    ratio = _follower_following_ratio(ig)
    if ratio is None:
        return None

    if ratio < 0.5:
        flags.append("more_following_than_followers")
        return 0.05

    if ratio < 1.0:
        return 0.1 + 0.2 * ratio

    if ratio < 2.0:
        flags.append("possible_follow_for_follow")
        return 0.3

    score = 0.5 + 0.5 * math.log(ratio / 2) / math.log(50)
    return min(1.0, max(0.0, score))


def _instagram_consistency(ig: PlatformMetrics, flags: list[str]) -> float | None:
    """Instagram consistency — variance matters more here than TikTok.

    On Instagram, consistent engagement = healthy follower relationship.
    Unlike TikTok where the algorithm causes natural variance, Instagram
    variance usually means content quality issues or follower decay.

    Uses coefficient of variation on like counts.
    """
    if not ig.view_counts and not ig.like_counts:
        return None

    # Prefer like counts for IG (more reliable than views)
    counts = ig.like_counts if ig.like_counts else ig.view_counts
    if len(counts) < 3:
        return None

    mean = statistics.mean(counts)
    if mean == 0:
        return None

    cv = statistics.stdev(counts) / mean

    # Instagram-specific curve (stricter than TikTok):
    # CV < 0.3 → 0.85-1.0 (very consistent)
    # CV 0.3-0.6 → 0.6-0.85 (normal)
    # CV 0.6-1.0 → 0.35-0.6 (inconsistent)
    # CV > 1.0 → 0.1-0.35 (problematic)
    score = 1.0 / (1.0 + cv * 1.5)  # Stricter penalty than TikTok
    return min(1.0, max(0.0, score))


def _instagram_authenticity(ig: PlatformMetrics, flags: list[str]) -> float | None:
    """Instagram authenticity scoring.

    Instagram-specific signals:
    1. Has bio and display name
    2. Reasonable follower/following ratio
    3. Meaningful post count relative to account age
    4. Comments exist (likes are easier to fake than comments)
    5. Verified status
    """
    signals: list[float] = []

    # 1. Has bio
    signals.append(1.0 if ig.bio and len(ig.bio) > 10 else 0.0)

    # 2. Has display name
    signals.append(1.0 if ig.display_name else 0.0)

    # 3. Reasonable follower/following ratio
    if ig.followers and ig.followers > 100:
        ratio = _follower_following_ratio(ig)
        if ratio is not None and 0.5 < ratio < 10000:
            signals.append(1.0)
        elif ratio is not None:
            signals.append(0.3)
        else:
            signals.append(0.5)
    else:
        signals.append(0.0)
        flags.append("very_low_followers")

    # 4. Meaningful post count
    post_count = ig.post_count or 0
    if post_count >= 50:
        signals.append(1.0)
    elif post_count >= 20:
        signals.append(0.8)
    elif post_count >= 5:
        signals.append(0.5)
    else:
        signals.append(0.1)
        flags.append("very_few_posts")

    # 5. Engagement rate in normal Instagram range
    er = ig.engagement_rate
    if er is not None:
        if 1.0 <= er <= 15.0:
            signals.append(1.0)
        elif er > 15.0:
            signals.append(0.2)
            flags.append("engagement_possibly_artificial")
        elif er > 0:
            signals.append(0.4)
        else:
            signals.append(0.1)
    else:
        signals.append(0.5)

    # 6. Verified
    signals.append(1.0 if ig.verified else 0.5)

    if not signals:
        return None

    return sum(signals) / len(signals)


# ======================================================================
# Fallback scoring (CSV data only, no platform-specific enrichment)
# ======================================================================


def _fallback_engagement(profile: CreatorProfile, flags: list[str]) -> float | None:
    """Fallback engagement scoring from CSV-provided data."""
    er = profile.primary_engagement_rate
    if er is None:
        return None

    if er > 15.0:
        flags.append("suspiciously_high_engagement")

    score = math.log(1 + er) / math.log(1 + 10)
    return min(1.0, max(0.0, score))


def _fallback_authenticity(profile: CreatorProfile, flags: list[str]) -> float | None:
    """Minimal authenticity check from CSV data."""
    platform = profile.tiktok or profile.instagram
    if platform is None:
        return None

    signals: list[float] = []
    signals.append(1.0 if platform.bio and len(platform.bio) > 10 else 0.3)
    signals.append(1.0 if platform.display_name else 0.3)

    content_count = platform.video_count or platform.post_count or 0
    if content_count >= 5:
        signals.append(1.0)
    elif content_count >= 1:
        signals.append(0.5)
    else:
        signals.append(0.2)

    return sum(signals) / len(signals) if signals else None
