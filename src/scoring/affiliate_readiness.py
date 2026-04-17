"""Affiliate Readiness Score — final composite ranking for outreach.

Combines content fit, TikTok trust factor, TikTok performance, TikTok
consistency, IG engagement, and authenticity into a single 0-100 score
that answers: "should we reach out to this creator?"

TRUST FACTOR is the #1 signal for non-perfect content fits. High TikTok
comments = audience trusts the creator = they'll buy what's recommended.
Comments indicate real back-and-forth conversation, not passive scrolling.

Weights:
  - Content Fit:          20%  (are they relevant to our product?)
  - TikTok Trust Factor:  25%  (comments = audience trust = purchase intent)
  - TikTok Performance:   20%  (views + like-to-view ratio + saves/shares)
  - TikTok Consistency:   15%  (regular posting with reliable reach?)
  - IG Engagement:        10%  (are their IG followers real and active?)
  - Authenticity:         10%  (organic signals, not botted)

Also includes an LLM recommendation for each creator — a 1-2 sentence
pitch explaining why they're a good (or bad) affiliate pick.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Default weights
W_CONTENT_FIT = 0.20
W_TIKTOK_TRUST = 0.25       # THE most important signal for affiliate success
W_TIKTOK_PERFORMANCE = 0.20
W_TIKTOK_CONSISTENCY = 0.15
W_IG_ENGAGEMENT = 0.10
W_AUTHENTICITY = 0.10


@dataclass
class AffiliateScore:
    """Final affiliate readiness score for a creator."""
    # Component scores (0.0 - 1.0)
    content_fit: float = 0.0
    tiktok_trust: float = 0.0
    tiktok_performance: float = 0.0
    tiktok_consistency: float = 0.0
    ig_engagement: float = 0.0
    authenticity: float = 0.0

    # Final composite (0 - 100)
    affiliate_score: float = 0.0

    # LLM recommendation
    recommendation: str = ""
    llm_score: float | None = None


def score_tiktok_trust(
    view_counts: list[int],
    comment_counts: list[int],
    like_counts: list[int],
) -> float:
    """Score TikTok trust factor — the #1 signal for affiliate success.

    High comments = audience actively engages with the creator in conversation.
    This means they trust the creator, feel connected, and will act on
    product recommendations. Passive viewers (views/likes) don't convert.

    IMPORTANT: Requires a meaningful sample size. A creator with 2 posts
    and high comments is NOT trustworthy data — it could be a fluke.
    Minimum 5 videos to get a real trust score.

    Signals:
      1. Average comments per video (raw volume)
      2. Comment-to-view ratio (engagement depth)
      3. Comment-to-like ratio (are engaged viewers actually talking?)

    Returns 0.0 - 1.0.
    """
    if not comment_counts or not view_counts:
        return 0.0

    # Penalize tiny sample sizes — need at least 5 videos for reliable trust signal
    sample_size = len(view_counts)
    if sample_size < 3:
        return 0.1  # Essentially no data
    sample_penalty = min(1.0, sample_size / 5.0)  # Ramps from 0.6 at 3 videos to 1.0 at 5+

    scores: list[float] = []

    # 1. Average comments per video — raw trust signal
    # More comments = more people feel compelled to respond
    avg_comments = statistics.mean(comment_counts) if comment_counts else 0

    # Log scale: 1 comment → 0.15, 5 → 0.45, 10 → 0.60, 25 → 0.75, 50 → 0.85, 100+ → 1.0
    if avg_comments > 0:
        comment_vol_score = math.log(1 + avg_comments) / math.log(1 + 100)
        scores.append(min(1.0, comment_vol_score))
    else:
        scores.append(0.0)

    # 2. Comment-to-view ratio — engagement depth
    # What % of viewers care enough to comment?
    comment_view_ratios = []
    for views, comments in zip(view_counts, comment_counts):
        if views and views > 0 and comments is not None:
            comment_view_ratios.append(comments / views * 100)

    if comment_view_ratios:
        median_cvr = statistics.median(comment_view_ratios)
        # TikTok benchmarks: 0.1% = low, 0.5% = decent, 1% = good, 2%+ = excellent
        cvr_score = min(1.0, median_cvr / 2.0)
        scores.append(cvr_score)

    # 3. Comment-to-like ratio — of people who liked, how many commented?
    # Higher = more conversational, deeper trust
    if like_counts and comment_counts:
        cl_ratios = []
        for likes, comments in zip(like_counts, comment_counts):
            if likes and likes > 0 and comments is not None:
                cl_ratios.append(comments / likes * 100)
        if cl_ratios:
            median_cl = statistics.median(cl_ratios)
            # 1% = low, 3% = decent, 5% = good, 10%+ = excellent
            cl_score = min(1.0, median_cl / 10.0)
            scores.append(cl_score)

    raw_score = sum(scores) / len(scores) if scores else 0.0
    return raw_score * sample_penalty  # Discount for small sample sizes


def score_tiktok_performance(
    view_counts: list[int],
    like_counts: list[int],
    comment_counts: list[int],
    share_counts: list[int],
    save_counts: list[int],
) -> float:
    """Score TikTok performance — the metrics that predict affiliate conversions.

    Signals:
      1. Median views — reliable reach per video
      2. Like-to-view ratio — are viewers engaged?
      3. Save+share ratio — the MONEY metric for affiliates
         Saves = "I want to buy this later"
         Shares = "others need to see this"
      4. Comment density — indicates community trust

    Returns 0.0 - 1.0.
    """
    if not view_counts or not like_counts:
        return 0.0

    # Penalize tiny TikTok accounts
    sample_size = len(view_counts)
    if sample_size < 3:
        return 0.1
    sample_penalty = min(1.0, sample_size / 5.0)

    scores: list[float] = []

    # 1. Median views — log scale (1K = 0.3, 10K = 0.6, 100K = 0.85, 1M = 1.0)
    median_views = statistics.median(view_counts) if view_counts else 0
    if median_views > 0:
        view_score = math.log10(max(1, median_views)) / 6.0  # log10(1M) = 6
        scores.append(min(1.0, view_score))
    else:
        scores.append(0.0)

    # 2. Like-to-view ratio (median across videos)
    ltv_ratios = []
    for views, likes in zip(view_counts, like_counts):
        if views and views > 0 and likes is not None:
            ltv_ratios.append(likes / views * 100)

    if ltv_ratios:
        median_ltv = statistics.median(ltv_ratios)
        # 1% → 0.25, 3% → 0.50, 5% → 0.65, 10% → 0.85, 15%+ → 1.0
        ltv_score = math.log(1 + median_ltv) / math.log(1 + 15)
        scores.append(min(1.0, ltv_score))

    # 3. Save + Share ratio (the affiliate money metric)
    if save_counts and share_counts and view_counts:
        intent_ratios = []
        for i, views in enumerate(view_counts):
            if views and views > 0:
                saves = save_counts[i] if i < len(save_counts) else 0
                shares = share_counts[i] if i < len(share_counts) else 0
                intent_ratios.append(((saves or 0) + (shares or 0)) / views * 100)

        if intent_ratios:
            median_intent = statistics.median(intent_ratios)
            # >0.5% = decent, >1% = good, >2% = excellent
            intent_score = min(1.0, median_intent / 2.0)
            scores.append(intent_score)

    # 4. Comment density — community trust signal
    if comment_counts and view_counts:
        comment_ratios = []
        for views, comments in zip(view_counts, comment_counts):
            if views and views > 0 and comments is not None:
                comment_ratios.append(comments / views * 100)

        if comment_ratios:
            median_comments = statistics.median(comment_ratios)
            # 0.1% → 0.3, 0.5% → 0.6, 1% → 0.8, 2%+ → 1.0
            comment_score = min(1.0, math.log(1 + median_comments * 10) / math.log(21))
            scores.append(comment_score)

    raw_score = sum(scores) / len(scores) if scores else 0.0
    return raw_score * sample_penalty


def score_tiktok_consistency(
    view_counts: list[int],
    video_timestamps: list[int],
) -> float:
    """Score TikTok posting consistency and view reliability.

    1. Floor reliability — P25/median views ratio
    2. Posting frequency — how often do they post?

    Returns 0.0 - 1.0.
    """
    if not view_counts or len(view_counts) < 3:
        return 0.3  # Can't assess, give neutral score

    scores: list[float] = []

    # 1. Floor reliability
    sorted_views = sorted(view_counts)
    median_views = statistics.median(sorted_views)
    p25 = sorted_views[len(sorted_views) // 4] if len(sorted_views) >= 4 else sorted_views[0]

    if median_views > 0:
        floor_ratio = p25 / median_views
        # Square root curve — generous for TikTok's natural variance
        floor_score = 0.3 + 0.7 * (floor_ratio ** 0.5)
        scores.append(min(1.0, floor_score))
    else:
        scores.append(0.0)

    # 2. Posting frequency
    if video_timestamps and len(video_timestamps) >= 2:
        timestamps = sorted(video_timestamps)
        gaps = [(timestamps[i + 1] - timestamps[i]) / 86400 for i in range(len(timestamps) - 1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 30

        if avg_gap <= 2:
            scores.append(1.0)    # Daily poster
        elif avg_gap <= 4:
            scores.append(0.85)   # Every few days
        elif avg_gap <= 7:
            scores.append(0.7)    # Weekly
        elif avg_gap <= 14:
            scores.append(0.5)    # Biweekly
        else:
            scores.append(0.25)   # Infrequent
    else:
        scores.append(0.5)  # Unknown frequency

    return sum(scores) / len(scores)


def score_ig_engagement(
    followers: int | None,
    avg_likes: float | None,
    avg_comments: float | None,
    engagement_rate: float | None,
) -> float:
    """Score Instagram engagement quality.

    Returns 0.0 - 1.0.
    """
    if engagement_rate is None or followers is None or followers == 0:
        return 0.3  # Can't assess

    # Cap insane ER values (API errors, tiny accounts)
    er = min(engagement_rate, 20.0)

    # Log curve tuned for IG benchmarks:
    # 1% → 0.30, 3% → 0.55, 6% → 0.75, 10% → 0.90, 15%+ → 1.0
    er_score = math.log(1 + er) / math.log(1 + 15)
    er_score = min(1.0, max(0.0, er_score))

    # Bonus for comments (harder to fake than likes)
    comment_bonus = 0.0
    if avg_comments and avg_likes and avg_likes > 0:
        comment_ratio = avg_comments / avg_likes
        # Comment-to-like ratio > 0.05 = engaged community
        if comment_ratio > 0.1:
            comment_bonus = 0.1
        elif comment_ratio > 0.05:
            comment_bonus = 0.05

    return min(1.0, er_score + comment_bonus)


def score_authenticity(
    ig_followers: int | None,
    ig_following: int | None,
    ig_verified: bool,
    ig_post_count: int | None,
    tiktok_like_counts: list[int] | None,
    tiktok_comment_counts: list[int] | None,
) -> float:
    """Score creator authenticity — is this a real, active creator?

    Returns 0.0 - 1.0.
    """
    signals: list[float] = []

    # IG follower/following ratio
    if ig_followers and ig_following and ig_following > 0:
        ratio = ig_followers / ig_following
        if ratio < 0.5:
            signals.append(0.1)  # More following than followers
        elif ratio < 1.5:
            signals.append(0.3)  # Follow-for-follow territory
        elif ratio < 10:
            signals.append(0.7)  # Normal
        else:
            signals.append(0.9)  # Strong organic

    # IG post count — established creator
    if ig_post_count:
        if ig_post_count >= 100:
            signals.append(1.0)
        elif ig_post_count >= 50:
            signals.append(0.8)
        elif ig_post_count >= 20:
            signals.append(0.6)
        else:
            signals.append(0.3)

    # IG verified
    signals.append(1.0 if ig_verified else 0.5)

    # TikTok organic engagement pattern
    if tiktok_like_counts and len(tiktok_like_counts) >= 3:
        non_zero = [lc for lc in tiktok_like_counts if lc and lc > 0]
        if len(non_zero) >= 2:
            cv = statistics.stdev(non_zero) / statistics.mean(non_zero) if statistics.mean(non_zero) > 0 else 0
            if 0.1 < cv < 5.0:
                signals.append(1.0)  # Natural variance
            elif cv <= 0.1:
                signals.append(0.3)  # Suspiciously uniform
            else:
                signals.append(0.6)

    # TikTok comments exist
    if tiktok_comment_counts:
        has_comments = sum(1 for c in tiktok_comment_counts if c and c > 0)
        signals.append(min(1.0, has_comments / max(1, len(tiktok_comment_counts))))

    return sum(signals) / len(signals) if signals else 0.5


def compute_affiliate_score(
    content_fit: float,
    tiktok_trust: float,
    tiktok_performance: float,
    tiktok_consistency: float,
    ig_engagement: float,
    authenticity: float,
) -> float:
    """Compute final weighted affiliate readiness score (0-100)."""
    raw = (
        content_fit * W_CONTENT_FIT
        + tiktok_trust * W_TIKTOK_TRUST
        + tiktok_performance * W_TIKTOK_PERFORMANCE
        + tiktok_consistency * W_TIKTOK_CONSISTENCY
        + ig_engagement * W_IG_ENGAGEMENT
        + authenticity * W_AUTHENTICITY
    )
    return round(raw * 100, 1)


async def get_llm_recommendation(
    handle: str,
    content_texts: list[str],
    bio: str,
    affiliate_score: float,
    tiktok_performance: float,
    ig_engagement: float,
    content_fit: float,
    ig_followers: int | None,
    tiktok_avg_views: float | None,
    campaign_brief: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "google/gemini-2.0-flash-001",
) -> tuple[float, str]:
    """Get LLM recommendation for why this creator is/isn't a good affiliate pick.

    Returns (score 0-10, recommendation string).
    """
    if not openrouter_api_key:
        return 0.0, ""

    content_sample = "\n---\n".join(content_texts[:10])
    if len(content_sample) > 2000:
        content_sample = content_sample[:2000] + "...[truncated]"

    followers_str = f"{ig_followers:,}" if ig_followers else "unknown"
    views_str = f"{tiktok_avg_views:,.0f}" if tiktok_avg_views else "unknown"

    prompt = f"""You are evaluating a social media creator as a potential affiliate partner for a probiotic gut health supplement.

**Campaign:** {campaign_brief or "Probiotic supplement for gut health targeting health-conscious women 18-45"}

**Creator: @{handle}**
- Bio: {bio or "(none)"}
- IG followers: {followers_str}
- TikTok avg views: {views_str}
- Our content-fit score: {content_fit:.0%}
- Our TikTok performance score: {tiktok_performance:.0%}
- Our IG engagement score: {ig_engagement:.0%}
- Our overall affiliate score: {affiliate_score}/100

**Their recent content:**
{content_sample}

**Task:** Give a 1-2 sentence recommendation about this creator as an affiliate partner. Be specific about WHY they're a good or bad fit — reference their actual content topics, audience, and metrics. If they're strong, say what makes them compelling. If they're weak, say what's missing.

Also rate them 0-10 for affiliate partnership readiness.

Respond with ONLY a JSON object:
{{"score": <0-10>, "recommendation": "<1-2 sentences>"}}"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": openrouter_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]

            parsed = json.loads(content)
            return float(parsed.get("score", 0)), parsed.get("recommendation", "")

    except Exception as e:
        logger.warning(f"LLM recommendation failed for @{handle}: {e}")
        return 0.0, ""


async def rank_creators(
    creators_data: list[dict],
    campaign_brief: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "google/gemini-2.0-flash-001",
    concurrency: int = 5,
) -> list[dict]:
    """Compute affiliate readiness scores and LLM recommendations for all creators.

    Args:
        creators_data: List of dicts, each with:
            - handle: str
            - content_fit_score: float (0-1)
            - tiktok_data: dict from yt-dlp (view_counts, like_counts, etc.)
            - ig_data: InstagramProfileData or None
            - bio: str
            - content_texts: list[str] (combined TT descriptions + IG captions)
        campaign_brief: Product description.
        openrouter_api_key: For LLM recommendations.
        openrouter_model: Model to use.
        concurrency: Max concurrent LLM calls.

    Returns:
        List of dicts with affiliate scores, sorted by score descending.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    total = len(creators_data)
    completed = 0

    async def _score_one(creator: dict) -> None:
        nonlocal completed
        handle = creator["handle"]
        tk = creator.get("tiktok_data") or {}
        ig = creator.get("ig_data")

        # --- Component Scores ---
        content_fit = creator.get("content_fit_score", 0.0)

        tiktok_trust = score_tiktok_trust(
            view_counts=tk.get("view_counts", []),
            comment_counts=tk.get("comment_counts", []),
            like_counts=tk.get("like_counts", []),
        )

        tiktok_perf = score_tiktok_performance(
            view_counts=tk.get("view_counts", []),
            like_counts=tk.get("like_counts", []),
            comment_counts=tk.get("comment_counts", []),
            share_counts=tk.get("share_counts", []),
            save_counts=tk.get("save_counts", []),
        )

        tiktok_consist = score_tiktok_consistency(
            view_counts=tk.get("view_counts", []),
            video_timestamps=[
                v["timestamp"] for v in tk.get("videos", [])
                if v.get("timestamp") is not None
            ] if tk.get("videos") else [],
        )

        ig_eng = score_ig_engagement(
            followers=ig.followers if ig else None,
            avg_likes=ig.avg_likes if ig else None,
            avg_comments=ig.avg_comments if ig else None,
            engagement_rate=ig.engagement_rate if ig else None,
        )

        auth = score_authenticity(
            ig_followers=ig.followers if ig else None,
            ig_following=ig.following if ig else None,
            ig_verified=ig.verified if ig else False,
            ig_post_count=ig.post_count if ig else None,
            tiktok_like_counts=tk.get("like_counts"),
            tiktok_comment_counts=tk.get("comment_counts"),
        )

        aff_score = compute_affiliate_score(
            content_fit=content_fit,
            tiktok_trust=tiktok_trust,
            tiktok_performance=tiktok_perf,
            tiktok_consistency=tiktok_consist,
            ig_engagement=ig_eng,
            authenticity=auth,
        )

        # --- LLM Recommendation ---
        llm_score = None
        recommendation = ""
        if openrouter_api_key:
            async with sem:
                llm_score, recommendation = await get_llm_recommendation(
                    handle=handle,
                    content_texts=creator.get("content_texts", []),
                    bio=creator.get("bio", ""),
                    affiliate_score=aff_score,
                    tiktok_performance=tiktok_perf,
                    ig_engagement=ig_eng,
                    content_fit=content_fit,
                    ig_followers=ig.followers if ig else None,
                    tiktok_avg_views=tk.get("avg_views"),
                    campaign_brief=campaign_brief,
                    openrouter_api_key=openrouter_api_key,
                    openrouter_model=openrouter_model,
                )

        completed += 1
        logger.info(
            f"[{completed}/{total}] @{handle}: "
            f"affiliate_score={aff_score}, "
            f"fit={content_fit:.2f}, trust={tiktok_trust:.2f}, "
            f"tt_perf={tiktok_perf:.2f}, tt_consist={tiktok_consist:.2f}, "
            f"ig_eng={ig_eng:.2f}"
        )

        results.append({
            "handle": handle,
            "affiliate_score": aff_score,
            "content_fit": round(content_fit, 3),
            "tiktok_trust": round(tiktok_trust, 3),
            "tiktok_performance": round(tiktok_perf, 3),
            "tiktok_consistency": round(tiktok_consist, 3),
            "ig_engagement": round(ig_eng, 3),
            "authenticity": round(auth, 3),
            "llm_score": llm_score,
            "recommendation": recommendation,
        })

    tasks = [_score_one(c) for c in creators_data]
    await asyncio.gather(*tasks)

    # Sort by affiliate_score descending
    results.sort(key=lambda r: r["affiliate_score"], reverse=True)

    # Add rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    logger.info(
        f"Affiliate ranking complete: {len(results)} creators scored. "
        f"Top score: {results[0]['affiliate_score'] if results else 'N/A'}"
    )
    return results
