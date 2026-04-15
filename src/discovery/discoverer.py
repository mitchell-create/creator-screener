"""Creator discovery engine.

Two modes:
  1. Direct handles — provide known creator handles to verify and score.
  2. AI-powered discovery — provide a campaign brief, LLM suggests creators
     in that niche, then we verify they exist and pull their data via yt-dlp.

Note: TikTok and Instagram hashtag page scraping is blocked by both Firecrawl
(403) and yt-dlp (requires app credentials). So discovery relies on the LLM's
knowledge of creators in various niches, verified against live profiles.
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx

from src.acquisition.tiktok_enricher import enrich_tiktok_profile

logger = logging.getLogger(__name__)


class CreatorDiscoverer:
    """Discovers potential affiliate creators via LLM suggestions + yt-dlp verification."""

    def __init__(
        self,
        openrouter_api_key: str = "",
        openrouter_model: str = "google/gemini-2.0-flash-001",
    ):
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model

    async def discover(
        self,
        campaign_brief: str = "",
        hashtags: list[str] | None = None,
        platforms: list[str] | None = None,
        max_creators: int = 50,
        min_followers: int = 1_000,
        max_followers: int = 10_000_000,
    ) -> list[dict]:
        """Discover creators matching a campaign brief or niche hashtags.

        Uses the LLM to suggest creators, then verifies them via yt-dlp.

        Returns:
            List of dicts with 'handle', 'platform', 'verified', and metadata.
        """
        if not self.openrouter_api_key:
            logger.error("OpenRouter API key required for discovery mode")
            return []

        if platforms is None:
            platforms = ["tiktok"]

        # Build the prompt
        niche_context = ""
        if hashtags:
            niche_context = f"\nRelevant hashtags/topics: {', '.join(hashtags)}"

        suggestions = await self._suggest_creators(
            campaign_brief=campaign_brief,
            niche_context=niche_context,
            platforms=platforms,
            count=max_creators,
            min_followers=min_followers,
            max_followers=max_followers,
        )

        if not suggestions:
            logger.warning("LLM returned no creator suggestions")
            return []

        logger.info(f"LLM suggested {len(suggestions)} creators. Verifying via yt-dlp...")

        # Verify TikTok handles exist and pull basic metrics
        verified = []
        if "tiktok" in platforms:
            tiktok_handles = [s["handle"] for s in suggestions if s.get("platform") == "tiktok"]
            verified_tk = await self._verify_tiktok_handles(tiktok_handles)
            verified.extend(verified_tk)

        # Instagram handles — store but can't verify without Firecrawl working
        if "instagram" in platforms:
            ig_handles = [s for s in suggestions if s.get("platform") == "instagram"]
            for s in ig_handles:
                s["verified"] = False  # Can't verify IG yet
                s["note"] = "Instagram handle — not verified (scraping unavailable)"
            verified.extend(ig_handles)

        logger.info(
            f"Discovery complete: {len(verified)} creators "
            f"({sum(1 for v in verified if v.get('verified'))} verified)"
        )
        return verified

    async def _suggest_creators(
        self,
        campaign_brief: str,
        niche_context: str,
        platforms: list[str],
        count: int,
        min_followers: int,
        max_followers: int,
    ) -> list[dict]:
        """Ask the LLM to suggest creators for a campaign."""
        platform_str = " and ".join(platforms)
        min_f = f"{min_followers:,}"
        max_f = f"{max_followers:,}"

        prompt = f"""You are an expert influencer marketing strategist. Suggest {count} real {platform_str} creators
who would be strong affiliate partners for the following campaign.

Campaign: {campaign_brief or 'General product promotion'}
{niche_context}

Requirements:
- Real creators who actively post on {platform_str} as of 2025-2026
- Follower range: {min_f} to {max_f}
- Focus on creators who do authentic product reviews, tutorials, or UGC-style content
- Prioritize creators known for genuine engagement over vanity metrics
- Mix of sizes: include some micro (10K-50K), mid (50K-500K), and if range allows, some larger creators
- Only suggest creators you are confident actually exist on the platform

Return a JSON array of objects, each with:
- "handle": the exact username (no @ prefix)
- "platform": "{platforms[0]}" or "{platforms[-1]}"
- "niche": brief description of their content focus
- "estimated_followers": rough follower count
- "reason": why they'd be a good fit for this campaign

Return ONLY the JSON array. No other text."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openrouter_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "max_tokens": 4000,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON from response
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]

                suggestions = json.loads(content)
                logger.info(f"LLM suggested {len(suggestions)} creators")
                return suggestions

        except Exception as e:
            logger.error(f"Failed to get creator suggestions: {e}")
            return []

    async def _verify_tiktok_handles(
        self,
        handles: list[str],
        concurrency: int = 3,
    ) -> list[dict]:
        """Verify TikTok handles exist and pull basic metrics."""
        sem = asyncio.Semaphore(concurrency)
        results = []

        async def _verify_one(handle: str) -> dict | None:
            async with sem:
                try:
                    data = await enrich_tiktok_profile(handle, video_count=5)
                    if data and data.get("videos"):
                        return {
                            "handle": handle,
                            "platform": "tiktok",
                            "verified": True,
                            "display_name": data.get("display_name"),
                            "avg_views": data.get("avg_views"),
                            "avg_likes": data.get("avg_likes"),
                            "avg_comments": data.get("avg_comments"),
                            "avg_shares": data.get("avg_shares"),
                            "avg_saves": data.get("avg_saves"),
                            "video_count_sampled": data.get("video_count_scraped"),
                        }
                    else:
                        logger.warning(f"@{handle}: no videos found or handle doesn't exist")
                        return {
                            "handle": handle,
                            "platform": "tiktok",
                            "verified": False,
                            "note": "No videos found or handle doesn't exist",
                        }
                except Exception as e:
                    logger.warning(f"@{handle}: verification failed: {e}")
                    return {
                        "handle": handle,
                        "platform": "tiktok",
                        "verified": False,
                        "note": f"Verification failed: {str(e)[:100]}",
                    }

        tasks = [_verify_one(h) for h in handles]
        gathered = await asyncio.gather(*tasks)
        return [r for r in gathered if r is not None]
