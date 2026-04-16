"""Content-fit filter for affiliate creator relevance.

Two-tier system:
  Tier A — Keyword scan (free, instant): scans TikTok video descriptions
           and IG post captions for health/wellness signal words.
  Tier B — LLM analysis (paid, Gemini): for ambiguous creators, asks the
           LLM to rate content relevance to the campaign product.

Designed for a probiotic gut health supplement, but the campaign brief
and keywords are configurable for any product.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Default campaign context — can be overridden
DEFAULT_CAMPAIGN_BRIEF = (
    "Probiotic supplement for gut health. Target audience: health-conscious women 18-45 "
    "who are interested in wellness routines, clean eating, supplements, fitness, "
    "and holistic health. The ideal affiliate creator regularly discusses health, "
    "nutrition, wellness routines, supplements, gut health, or body care in their content."
)

# --- Keyword Tiers ---
# Each keyword has a weight. We scan all content and compute a weighted score.

STRONG_KEYWORDS: dict[str, float] = {
    # Gut health specific
    "probiotic": 3.0,
    "prebiotic": 3.0,
    "gut health": 3.0,
    "gut-health": 3.0,
    "digestive": 2.5,
    "digestion": 2.5,
    "microbiome": 3.0,
    "bloating": 2.5,
    "bloat": 2.0,
    "fermented": 2.0,
    "kombucha": 2.0,
    "fiber supplement": 2.5,
    "leaky gut": 3.0,
    "ibs": 2.5,
    "gut flora": 3.0,
    "gut-friendly": 2.5,
    "gut healing": 3.0,
}

MEDIUM_KEYWORDS: dict[str, float] = {
    # Health & wellness
    "supplement": 1.5,
    "supplements": 1.5,
    "vitamin": 1.5,
    "vitamins": 1.5,
    "wellness": 1.5,
    "nutrition": 1.5,
    "nutritionist": 1.5,
    "dietitian": 1.5,
    "healthy eating": 1.5,
    "clean eating": 1.5,
    "holistic": 1.5,
    "organic": 1.0,
    "whole food": 1.5,
    "whole foods": 1.0,
    "plant-based": 1.0,
    "plant based": 1.0,
    "meal prep": 1.0,
    "smoothie": 1.0,
    "green juice": 1.5,
    "detox": 1.0,
    "cleanse": 1.0,
    "anti-inflammatory": 2.0,
    "inflammation": 1.5,
    "immune": 1.0,
    "immunity": 1.0,
    "morning routine": 1.0,
    "wellness routine": 1.5,
    "self care": 1.0,
    "self-care": 1.0,
    "health journey": 1.5,
    "fitness": 1.0,
    "workout": 0.8,
    "protein": 1.0,
    "collagen": 1.5,
    "turmeric": 1.5,
    "adaptogens": 2.0,
    "functional medicine": 2.0,
    "naturopath": 2.0,
    "integrative": 1.5,
    "hormone": 1.5,
    "hormones": 1.5,
    "cortisol": 1.5,
    "adrenal": 1.5,
    "thyroid": 1.5,
    "autoimmune": 2.0,
    "gluten free": 1.0,
    "gluten-free": 1.0,
    "dairy free": 1.0,
    "dairy-free": 1.0,
    "food sensitivity": 2.0,
    "blood sugar": 1.5,
    "weight loss": 0.8,
    "macro": 0.8,
    "macros": 0.8,
    "healing": 1.0,
    "nourish": 1.0,
}

WEAK_KEYWORDS: dict[str, float] = {
    # Lifestyle adjacent — alone not enough, but supportive
    # NOTE: skincare, recipe, food by themselves do NOT indicate health/wellness.
    # A food blogger who just eats food is not a fit. A skincare-only creator is not a fit.
    # These only matter when combined with actual health signals.
    "recipe": 0.1,
    "recipes": 0.1,
    "skincare": 0.1,     # Downweighted — pure skincare ≠ health/wellness
    "skin care": 0.1,
    "hair health": 0.1,  # Downweighted — pure hair ≠ health/wellness
    "natural": 0.2,
    "balance": 0.2,
    "energy": 0.2,
    "body": 0.1,
    "mindful": 0.4,
    "meditation": 0.3,
    "yoga": 0.3,
    "pilates": 0.3,
    "active": 0.1,
    "lifestyle": 0.1,
    "routine": 0.1,
    "clean": 0.1,
    "healthy": 0.3,
    "health": 0.3,
}

# Combine all keywords
ALL_KEYWORDS: dict[str, float] = {**STRONG_KEYWORDS, **MEDIUM_KEYWORDS, **WEAK_KEYWORDS}

# Thresholds — NO auto-pass. Every creator goes through LLM review.
# Keywords alone can't distinguish "hair supplements" from "gut supplements"
# or "cat nutrition" from "human nutrition." The LLM catches these nuances
# and costs <$0.001 per creator, so there's no reason to skip it.
KEYWORD_PASS_THRESHOLD = 999.0  # Effectively disabled — nobody auto-passes
KEYWORD_FAIL_THRESHOLD = 0.5    # Below this → auto-fail (truly zero health signal)
# Everything above fail threshold → sent to LLM for proper judgment


@dataclass
class ContentFitResult:
    """Result of content-fit analysis for a creator."""
    score: float = 0.0           # 0.0 - 1.0 normalized fit score
    passed: bool = False         # Whether the creator passes the content-fit filter
    method: str = ""             # "keyword_pass", "keyword_fail", "llm"
    reason: str = ""             # Human-readable explanation
    keyword_score: float = 0.0   # Raw weighted keyword score
    keyword_matches: dict[str, int] = field(default_factory=dict)  # keyword -> count
    llm_score: float | None = None  # LLM score (0-10) if Tier B was used
    llm_reasoning: str = ""      # LLM explanation if Tier B was used


def scan_keywords(content_texts: list[str]) -> tuple[float, dict[str, int]]:
    """Scan a list of content texts for health/wellness keywords.

    Args:
        content_texts: List of video descriptions / post captions.

    Returns:
        Tuple of (weighted_score, keyword_match_counts).
    """
    if not content_texts:
        return 0.0, {}

    # Combine all content into one lowercase string for matching
    combined = " ".join(content_texts).lower()

    matches: dict[str, int] = {}
    total_score = 0.0

    for keyword, weight in ALL_KEYWORDS.items():
        # Use word boundary matching for short keywords to avoid false positives
        if len(keyword) <= 4:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            count = len(re.findall(pattern, combined))
        else:
            count = combined.count(keyword.lower())

        if count > 0:
            matches[keyword] = count
            # Diminishing returns: first match gets full weight, subsequent get less
            total_score += weight * (1 + min(count - 1, 4) * 0.25)

    return total_score, matches


def classify_keyword_result(
    keyword_score: float,
    matches: dict[str, int],
) -> ContentFitResult:
    """Classify a creator based on keyword scan results.

    Returns a ContentFitResult with method set to keyword_pass, keyword_fail,
    or keyword_ambiguous (needs LLM).
    """
    result = ContentFitResult(
        keyword_score=keyword_score,
        keyword_matches=matches,
    )

    # Check for strong gut-health signals
    strong_matches = {k: v for k, v in matches.items() if k in STRONG_KEYWORDS}
    medium_matches = {k: v for k, v in matches.items() if k in MEDIUM_KEYWORDS}

    if keyword_score >= KEYWORD_PASS_THRESHOLD:
        # Clear health/wellness creator
        result.passed = True
        result.method = "keyword_pass"
        result.score = min(1.0, keyword_score / 10.0)

        top_keywords = sorted(matches.items(), key=lambda x: ALL_KEYWORDS.get(x[0], 0) * x[1], reverse=True)[:5]
        top_str = ", ".join(f"{k}({v})" for k, v in top_keywords)
        result.reason = f"Health/wellness content confirmed — top signals: {top_str}"

    elif keyword_score < KEYWORD_FAIL_THRESHOLD:
        # No health signal at all
        result.passed = False
        result.method = "keyword_fail"
        result.score = keyword_score / 10.0

        if matches:
            result.reason = f"Minimal health signals (score {keyword_score:.1f}) — likely not wellness-focused"
        else:
            result.reason = "No health/wellness keywords found in any content"

    else:
        # Ambiguous — needs LLM review
        result.method = "keyword_ambiguous"
        result.score = keyword_score / 10.0
        result.reason = f"Some health signals (score {keyword_score:.1f}) — needs LLM review"

    return result


async def llm_content_fit(
    content_texts: list[str],
    bio: str = "",
    campaign_brief: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "google/gemini-2.0-flash-001",
) -> tuple[float, str]:
    """Ask the LLM to rate content relevance to the campaign.

    Args:
        content_texts: Combined video descriptions + post captions.
        bio: Creator's bio text.
        campaign_brief: Product/campaign description.
        openrouter_api_key: API key for OpenRouter.
        openrouter_model: Model to use.

    Returns:
        Tuple of (score 0-10, reasoning string).
    """
    if not openrouter_api_key:
        return 0.0, "No API key for LLM content-fit scoring"

    if not campaign_brief:
        campaign_brief = DEFAULT_CAMPAIGN_BRIEF

    # Build content sample — limit to avoid token bloat
    content_sample = "\n---\n".join(content_texts[:15])
    if len(content_sample) > 3000:
        content_sample = content_sample[:3000] + "...[truncated]"

    prompt = f"""You are evaluating whether a social media creator is a good fit to promote a PROBIOTIC GUT HEALTH SUPPLEMENT as an affiliate partner.

**Product/Campaign:**
{campaign_brief}

**Creator's Bio:**
{bio or "(no bio available)"}

**Recent Content (video descriptions & post captions):**
{content_sample}

**IMPORTANT DISTINCTIONS — read carefully:**

GOOD FIT examples:
- Creators who discuss physical health, nutrition, supplements, wellness routines, gut health, digestive issues, healthy eating habits, fitness + nutrition, holistic health, functional medicine
- GLP-1/weight loss medication creators ARE a good fit (gut health is a common concern for GLP-1 users)
- Fitness creators who also discuss nutrition, supplements, or wellness routines
- Lifestyle creators who regularly discuss their health journey, supplement stacks, or wellness habits

NOT A FIT examples:
- Pure skincare/beauty creators who ONLY discuss topical products (serums, makeup, hair products) without discussing internal health, nutrition, or supplements
- Pure food/recipe creators who just eat or cook food without discussing health benefits, nutrition, or dietary wellness
- Pet health creators (cat nutrition, dog supplements, etc.)
- Fashion, travel, home decor, book, or entertainment-only creators
- Creators who mention "healthy" occasionally but whose content is clearly not health-focused

The key question: Does this creator regularly talk about PHYSICAL HEALTH, NUTRITION, or WELLNESS in a way that would make a probiotic supplement feel natural in their content?

Rate on a scale of 0-10:
- 0-3: Not a fit (content doesn't meaningfully cover health/wellness/nutrition)
- 4-5: Borderline (some health adjacent content but not core to their brand)
- 6-7: Good fit (health/wellness is a regular theme alongside other content)
- 8-10: Excellent fit (health, nutrition, or supplements are central to their content)

Respond with ONLY a JSON object:
{{"score": <0-10>, "reasoning": "<1-2 sentence explanation>"}}"""

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

            # Parse JSON response
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]

            parsed = json.loads(content)
            score = float(parsed.get("score", 0))
            reasoning = parsed.get("reasoning", "")
            return score, reasoning

    except Exception as e:
        logger.warning(f"LLM content-fit scoring failed: {e}")
        return 0.0, f"LLM error: {str(e)[:100]}"


async def evaluate_content_fit(
    tiktok_descriptions: list[str],
    ig_captions: list[str],
    bio: str = "",
    campaign_brief: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "google/gemini-2.0-flash-001",
) -> ContentFitResult:
    """Full content-fit evaluation: keyword scan → optional LLM.

    Args:
        tiktok_descriptions: Video description texts from TikTok.
        ig_captions: Post caption texts from Instagram.
        bio: Creator's bio.
        campaign_brief: Product/campaign description.
        openrouter_api_key: For LLM tier.
        openrouter_model: Model for LLM tier.

    Returns:
        ContentFitResult with score, pass/fail, and reasoning.
    """
    all_content = tiktok_descriptions + ig_captions

    # Also include bio in keyword scan
    if bio:
        all_content_with_bio = [bio] + all_content
    else:
        all_content_with_bio = all_content

    # --- Tier A: Keyword scan ---
    keyword_score, matches = scan_keywords(all_content_with_bio)
    result = classify_keyword_result(keyword_score, matches)

    logger.info(
        f"Keyword scan: score={keyword_score:.1f}, "
        f"matches={len(matches)}, method={result.method}"
    )

    # If keyword scan is decisive, return immediately
    if result.method in ("keyword_pass", "keyword_fail"):
        return result

    # --- Tier B: LLM analysis for ambiguous cases ---
    if openrouter_api_key:
        logger.info("Ambiguous content — sending to LLM for analysis")
        llm_score, llm_reasoning = await llm_content_fit(
            content_texts=all_content,
            bio=bio,
            campaign_brief=campaign_brief,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
        )

        result.llm_score = llm_score
        result.llm_reasoning = llm_reasoning
        result.method = "llm"

        # LLM score 0-10 → normalize to 0-1
        result.score = llm_score / 10.0

        if llm_score >= 6.0:
            result.passed = True
            result.reason = f"LLM approved ({llm_score}/10): {llm_reasoning}"
        else:
            result.passed = False
            result.reason = f"LLM rejected ({llm_score}/10): {llm_reasoning}"
    else:
        # No LLM key — use keyword score as best effort
        result.passed = keyword_score >= 2.5  # Middle ground
        result.reason = f"Ambiguous (score {keyword_score:.1f}) — no LLM key for deeper analysis"

    return result


async def batch_evaluate_content_fit(
    creators_content: list[dict],
    campaign_brief: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "google/gemini-2.0-flash-001",
    concurrency: int = 3,
) -> dict[str, ContentFitResult]:
    """Evaluate content fit for a batch of creators.

    Args:
        creators_content: List of dicts with keys:
            - handle: str
            - tiktok_descriptions: list[str]
            - ig_captions: list[str]
            - bio: str
        campaign_brief: Product/campaign description.
        openrouter_api_key: For LLM tier.
        openrouter_model: Model for LLM tier.
        concurrency: Max concurrent LLM calls.

    Returns:
        Dict mapping handle -> ContentFitResult.
    """
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, ContentFitResult] = {}
    total = len(creators_content)
    completed = 0

    async def _evaluate_one(creator: dict) -> None:
        nonlocal completed
        handle = creator["handle"]

        # Keyword scan is instant — no semaphore needed
        # But LLM calls need rate limiting
        async with sem:
            result = await evaluate_content_fit(
                tiktok_descriptions=creator.get("tiktok_descriptions", []),
                ig_captions=creator.get("ig_captions", []),
                bio=creator.get("bio", ""),
                campaign_brief=campaign_brief,
                openrouter_api_key=openrouter_api_key,
                openrouter_model=openrouter_model,
            )
            results[handle] = result
            completed += 1

            status = "✓ pass" if result.passed else "✗ fail"
            logger.info(
                f"[{completed}/{total}] @{handle}: {status} "
                f"(method={result.method}, score={result.score:.2f})"
            )

    tasks = [_evaluate_one(c) for c in creators_content]
    await asyncio.gather(*tasks)

    passed = sum(1 for r in results.values() if r.passed)
    failed = total - passed
    methods = {}
    for r in results.values():
        methods[r.method] = methods.get(r.method, 0) + 1

    logger.info(
        f"Content-fit complete: {passed} passed, {failed} failed. "
        f"Methods: {methods}"
    )
    return results
