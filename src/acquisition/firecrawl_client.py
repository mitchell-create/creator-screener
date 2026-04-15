"""Firecrawl-based scraper for TikTok and Instagram creator profiles.

Uses the Firecrawl API to scrape rendered pages and extract structured
creator metrics -- no proxy rotation or browser automation needed.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"
_TIMEOUT = 30.0


@dataclass
class TikTokProfile:
    """Structured data extracted from a TikTok profile page."""

    handle: str
    display_name: str | None = None
    bio: str | None = None
    followers: int | None = None
    following: int | None = None
    total_likes: int | None = None
    verified: bool = False
    video_count: int | None = None
    # Per-video stats from recent posts (scraped from profile page)
    recent_videos: list[TikTokVideoMeta] = field(default_factory=list)

    @property
    def follower_following_ratio(self) -> float | None:
        if self.followers and self.following and self.following > 0:
            return self.followers / self.following
        return None

    @property
    def avg_likes_per_video(self) -> float | None:
        if self.total_likes and self.video_count and self.video_count > 0:
            return self.total_likes / self.video_count
        return None


@dataclass
class TikTokVideoMeta:
    """Basic metadata for a video visible on the profile page."""

    video_id: str = ""
    description: str = ""
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None


@dataclass
class InstagramProfile:
    """Structured data extracted from an Instagram profile page."""

    handle: str
    display_name: str | None = None
    bio: str | None = None
    followers: int | None = None
    following: int | None = None
    post_count: int | None = None
    verified: bool = False
    is_business: bool = False
    # Engagement from visible posts
    recent_posts: list[InstagramPostMeta] = field(default_factory=list)

    @property
    def follower_following_ratio(self) -> float | None:
        if self.followers and self.following and self.following > 0:
            return self.followers / self.following
        return None

    @property
    def avg_engagement_rate(self) -> float | None:
        """Average engagement rate across recent visible posts."""
        if not self.recent_posts or not self.followers or self.followers == 0:
            return None
        rates = []
        for post in self.recent_posts:
            interactions = (post.likes or 0) + (post.comments or 0)
            if interactions > 0:
                rates.append(interactions / self.followers * 100)
        return sum(rates) / len(rates) if rates else None


@dataclass
class InstagramPostMeta:
    """Basic metadata for a post visible on the profile page."""

    post_id: str = ""
    description: str = ""
    likes: int | None = None
    comments: int | None = None


class FirecrawlScraper:
    """Scrapes TikTok and Instagram profiles via Firecrawl API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _scrape_url(self, url: str) -> str | None:
        """Scrape a URL via Firecrawl and return the markdown content."""
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{_FIRECRAWL_BASE}/scrape",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "waitFor": 3000,  # Wait for JS rendering
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", {}).get("markdown")
        except httpx.HTTPStatusError as e:
            logger.warning(f"Firecrawl HTTP error for {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Firecrawl error for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # TikTok
    # ------------------------------------------------------------------

    async def scrape_tiktok_profile(self, handle: str) -> TikTokProfile | None:
        """Scrape a TikTok creator's profile page and extract metrics."""
        clean_handle = handle.lstrip("@")
        url = f"https://www.tiktok.com/@{clean_handle}"
        logger.info(f"Scraping TikTok profile: {url}")

        markdown = await self._scrape_url(url)
        if not markdown:
            return None

        return self._parse_tiktok_profile(clean_handle, markdown)

    def _parse_tiktok_profile(self, handle: str, markdown: str) -> TikTokProfile:
        """Parse TikTok profile metrics from Firecrawl markdown output."""
        profile = TikTokProfile(handle=handle)

        # TikTok profile pages render stats in predictable patterns
        # Try multiple patterns since Firecrawl markdown varies

        # Follower count patterns: "1.2M Followers", "12.5K Followers", "1234 Followers"
        followers = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*Follower")
        if followers is not None:
            profile.followers = followers

        # Following count
        following = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*Following")
        if following is not None:
            profile.following = following

        # Total likes
        likes = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*Like")
        if likes is not None:
            profile.total_likes = likes

        # Display name - usually the first heading or bold text
        name_match = re.search(r"#\s*(.+?)[\n|]", markdown)
        if name_match:
            profile.display_name = name_match.group(1).strip()

        # Bio - text after the stats line, before video grid
        bio_match = re.search(
            r"(?:Likes?|Following|Followers?)\s*\n+(.+?)(?:\n\n|\n#|\n\*\*|$)",
            markdown,
            re.DOTALL,
        )
        if bio_match:
            bio_text = bio_match.group(1).strip()
            if len(bio_text) < 500:  # Sanity check
                profile.bio = bio_text

        # Verified badge
        if "✓" in markdown or "verified" in markdown.lower():
            profile.verified = True

        # Extract video metadata from the profile grid
        profile.recent_videos = self._parse_tiktok_videos(markdown)
        if profile.recent_videos:
            profile.video_count = len(profile.recent_videos)

        logger.info(
            f"TikTok @{handle}: "
            f"followers={profile.followers}, following={profile.following}, "
            f"likes={profile.total_likes}, videos={profile.video_count}"
        )
        return profile

    def _parse_tiktok_videos(self, markdown: str) -> list[TikTokVideoMeta]:
        """Extract video metadata from the profile page markdown."""
        videos = []
        # TikTok video entries often show view counts like "1.2M views", "45.6K views"
        view_pattern = re.compile(
            r"([\d,.]+[KkMmBb]?)\s*(?:views?|plays?)",
            re.IGNORECASE,
        )
        for match in view_pattern.finditer(markdown):
            views = self._parse_abbreviated_number(match.group(1))
            if views is not None:
                videos.append(TikTokVideoMeta(views=views))

        return videos[:20]  # Cap at 20 most visible

    # ------------------------------------------------------------------
    # Instagram
    # ------------------------------------------------------------------

    async def scrape_instagram_profile(self, handle: str) -> InstagramProfile | None:
        """Scrape an Instagram creator's profile page and extract metrics."""
        clean_handle = handle.lstrip("@")
        url = f"https://www.instagram.com/{clean_handle}/"
        logger.info(f"Scraping Instagram profile: {url}")

        markdown = await self._scrape_url(url)
        if not markdown:
            return None

        return self._parse_instagram_profile(clean_handle, markdown)

    def _parse_instagram_profile(self, handle: str, markdown: str) -> InstagramProfile:
        """Parse Instagram profile metrics from Firecrawl markdown output."""
        profile = InstagramProfile(handle=handle)

        # Instagram stats: "1,234 posts", "12.5K followers", "567 following"
        posts = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*posts?")
        if posts is not None:
            profile.post_count = posts

        followers = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*followers?")
        if followers is not None:
            profile.followers = followers

        following = self._extract_metric(markdown, r"([\d,.]+[KkMmBb]?)\s*following")
        if following is not None:
            profile.following = following

        # Display name
        name_match = re.search(r"#\s*(.+?)[\n|]", markdown)
        if name_match:
            profile.display_name = name_match.group(1).strip()

        # Bio
        bio_match = re.search(
            r"(?:posts?|followers?|following)\s*\n+(.+?)(?:\n\n|\n#|\n\*\*|$)",
            markdown,
            re.DOTALL | re.IGNORECASE,
        )
        if bio_match:
            bio_text = bio_match.group(1).strip()
            if len(bio_text) < 500:
                profile.bio = bio_text

        # Verified
        if "✓" in markdown or "Verified" in markdown:
            profile.verified = True

        # Extract post engagement from visible posts
        profile.recent_posts = self._parse_instagram_posts(markdown)

        logger.info(
            f"Instagram @{handle}: "
            f"followers={profile.followers}, following={profile.following}, "
            f"posts={profile.post_count}"
        )
        return profile

    def _parse_instagram_posts(self, markdown: str) -> list[InstagramPostMeta]:
        """Extract post metadata from visible posts on the profile page."""
        posts = []
        # Instagram posts show like counts: "1,234 likes", "12.5K likes"
        like_pattern = re.compile(
            r"([\d,.]+[KkMmBb]?)\s*likes?",
            re.IGNORECASE,
        )
        for match in like_pattern.finditer(markdown):
            likes = self._parse_abbreviated_number(match.group(1))
            if likes is not None:
                posts.append(InstagramPostMeta(likes=likes))

        return posts[:12]  # Cap at visible grid

    # ------------------------------------------------------------------
    # Discovery: scrape hashtag pages to find creators
    # ------------------------------------------------------------------

    async def discover_tiktok_creators(self, hashtag: str) -> list[str]:
        """Discover TikTok creator handles from a hashtag page.

        Returns a list of @handles found on the hashtag page.
        """
        clean_tag = hashtag.lstrip("#")
        url = f"https://www.tiktok.com/tag/{clean_tag}"
        logger.info(f"Discovering TikTok creators for #{clean_tag}")

        markdown = await self._scrape_url(url)
        if not markdown:
            return []

        # Extract @handles from the markdown
        handles = set()
        for match in re.finditer(r"@([\w.]+)", markdown):
            handle = match.group(1)
            # Filter out common non-handle matches
            if len(handle) >= 2 and not handle.startswith("."):
                handles.add(f"@{handle}")

        logger.info(f"Found {len(handles)} creators for #{clean_tag}")
        return sorted(handles)

    async def discover_instagram_creators(self, hashtag: str) -> list[str]:
        """Discover Instagram creator handles from a hashtag page.

        Returns a list of @handles found on the hashtag page.
        """
        clean_tag = hashtag.lstrip("#")
        url = f"https://www.instagram.com/explore/tags/{clean_tag}/"
        logger.info(f"Discovering Instagram creators for #{clean_tag}")

        markdown = await self._scrape_url(url)
        if not markdown:
            return []

        handles = set()
        for match in re.finditer(r"@([\w.]+)", markdown):
            handle = match.group(1)
            if len(handle) >= 2 and not handle.startswith("."):
                handles.add(f"@{handle}")

        logger.info(f"Found {len(handles)} creators for #{clean_tag}")
        return sorted(handles)

    async def discover_creators_multi(
        self,
        hashtags: list[str],
        platforms: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Discover creators across multiple hashtags and platforms.

        Args:
            hashtags: List of hashtags to search (with or without #).
            platforms: List of platforms ("tiktok", "instagram"). Defaults to both.

        Returns:
            Dict mapping platform -> deduplicated list of handles.
        """
        if platforms is None:
            platforms = ["tiktok", "instagram"]

        results: dict[str, set[str]] = {p: set() for p in platforms}

        tasks = []
        task_meta = []  # Track which platform each task belongs to
        for tag in hashtags:
            if "tiktok" in platforms:
                tasks.append(self.discover_tiktok_creators(tag))
                task_meta.append("tiktok")
            if "instagram" in platforms:
                tasks.append(self.discover_instagram_creators(tag))
                task_meta.append("instagram")

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for platform, result in zip(task_meta, gathered):
            if isinstance(result, Exception):
                logger.warning(f"Discovery failed for {platform}: {result}")
            else:
                results[platform].update(result)

        return {p: sorted(handles) for p, handles in results.items()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _extract_metric(self, text: str, pattern: str) -> int | None:
        """Extract a numeric metric using a regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return self._parse_abbreviated_number(match.group(1))
        return None

    @staticmethod
    def _parse_abbreviated_number(text: str) -> int | None:
        """Parse abbreviated numbers like '1.2M', '45.6K', '1,234'."""
        if not text:
            return None
        text = text.strip().replace(",", "")
        multiplier = 1
        suffix = text[-1].upper()
        if suffix == "K":
            multiplier = 1_000
            text = text[:-1]
        elif suffix == "M":
            multiplier = 1_000_000
            text = text[:-1]
        elif suffix == "B":
            multiplier = 1_000_000_000
            text = text[:-1]
        try:
            return int(float(text) * multiplier)
        except ValueError:
            return None
