"""Instagram profile enrichment via RapidAPI (Instagram Scraper Stable API).

Replaces the Firecrawl-based scraper which was blocked by 403s.
Uses two endpoints:
  1. Account Data — profile metadata (followers, following, bio, verified)
  2. User Posts — recent posts with engagement (likes, comments)

API: https://rapidapi.com/thetechguy32744/api/instagram-scraper-stable-api
Host: instagram-scraper-stable-api.p.rapidapi.com

Endpoints use POST with application/x-www-form-urlencoded body.

Account Data response: flat object with 58+ keys including:
  username, full_name, biography, follower_count, following_count,
  media_count, is_verified, is_private, is_business, profile_pic_url,
  external_url, category, pk, id

User Posts response: { posts: [ { node: { ...81 keys } } ] }
  Per-post node includes: like_count, comment_count, taken_at (unix),
  media_type (1=image, 2=video, 8=carousel), caption: { text: "..." },
  view_count, code, pk, id
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_RAPIDAPI_BASE = "https://instagram-scraper-stable-api.p.rapidapi.com"
_RAPIDAPI_HOST = "instagram-scraper-stable-api.p.rapidapi.com"


@dataclass
class InstagramPostData:
    """Engagement data for a single Instagram post."""
    post_id: str = ""
    caption: str = ""
    likes: int = 0
    comments: int = 0
    views: int | None = None
    media_type: str = ""  # "image", "video", "carousel"
    timestamp: int | None = None


@dataclass
class InstagramProfileData:
    """Structured profile data from RapidAPI."""
    handle: str = ""
    display_name: str | None = None
    bio: str | None = None
    followers: int | None = None
    following: int | None = None
    post_count: int | None = None
    verified: bool = False
    is_business: bool = False
    is_private: bool = False
    profile_pic_url: str | None = None
    external_url: str | None = None
    category: str | None = None

    # Computed from recent posts
    recent_posts: list[InstagramPostData] = field(default_factory=list)
    avg_likes: float | None = None
    avg_comments: float | None = None
    engagement_rate: float | None = None


class InstagramEnricher:
    """Enriches Instagram profiles via RapidAPI Instagram Scraper Stable API."""

    def __init__(
        self,
        rapidapi_key: str,
        rapidapi_host: str = _RAPIDAPI_HOST,
    ):
        self.rapidapi_key = rapidapi_key
        self.rapidapi_host = rapidapi_host
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "x-rapidapi-key": self.rapidapi_key,
            "x-rapidapi-host": self.rapidapi_host,
            "Content-Type": "application/x-www-form-urlencoded",
        }

    async def enrich_profile(
        self,
        handle: str,
        post_count: int = 12,
        delay_range: tuple[float, float] = (0.5, 2.0),
    ) -> InstagramProfileData | None:
        """Fetch full profile data + recent post engagement for an IG handle.

        Args:
            handle: Instagram username (with or without @).
            post_count: Number of recent posts to fetch for engagement calc.
            delay_range: Random delay between API calls.

        Returns:
            InstagramProfileData with engagement metrics, or None on failure.
        """
        clean_handle = handle.lstrip("@")

        # Random delay to stay within rate limits
        delay = random.uniform(*delay_range)
        await asyncio.sleep(delay)

        # Step 1: Get profile info via Account Data endpoint
        profile = await self._fetch_account_data(clean_handle)
        if profile is None:
            return None

        if profile.is_private:
            logger.info(f"IG @{clean_handle}: private account — skipping post fetch")
            return profile

        # Small delay between calls
        await asyncio.sleep(random.uniform(0.3, 1.0))

        # Step 2: Get recent posts for engagement via User Posts endpoint
        posts = await self._fetch_user_posts(clean_handle, count=post_count)
        if posts:
            profile.recent_posts = posts
            self._compute_engagement(profile)

        return profile

    async def _fetch_account_data(self, handle: str) -> InstagramProfileData | None:
        """Fetch profile metadata via POST /ig_get_fb_profile.php.

        Response is a flat JSON object with keys like:
          username, full_name, biography, follower_count, following_count,
          media_count, is_verified, is_private, is_business, profile_pic_url,
          external_url, category
        """
        client = await self._get_client()

        try:
            resp = await client.post(
                f"{_RAPIDAPI_BASE}/ig_get_fb_profile.php",
                headers=self._headers(),
                data={
                    "username_or_url": handle,
                    "data": "basic",
                },
            )
            resp.raise_for_status()
            user = resp.json()

            # Response is a flat object — no wrapper
            profile = InstagramProfileData(
                handle=user.get("username", handle),
                display_name=user.get("full_name"),
                bio=user.get("biography"),
                followers=_safe_int(user.get("follower_count")),
                following=_safe_int(user.get("following_count")),
                post_count=_safe_int(user.get("media_count")),
                verified=bool(user.get("is_verified", False)),
                is_business=bool(user.get("is_business", False)),
                is_private=bool(user.get("is_private", False)),
                profile_pic_url=user.get("profile_pic_url"),
                external_url=user.get("external_url"),
                category=user.get("category"),
            )

            logger.info(
                f"IG @{handle}: followers={profile.followers}, "
                f"following={profile.following}, posts={profile.post_count}, "
                f"verified={profile.verified}, private={profile.is_private}"
            )
            return profile

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"IG @{handle}: profile not found (404)")
            elif e.response.status_code == 429:
                logger.warning(f"IG @{handle}: rate limited (429) — back off")
            else:
                logger.warning(f"IG @{handle}: API error {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"IG @{handle}: fetch failed: {e}")
            return None

    async def _fetch_user_posts(
        self,
        handle: str,
        count: int = 12,
    ) -> list[InstagramPostData]:
        """Fetch recent posts via POST /get_ig_user_posts.php.

        Response format: { posts: [ { node: { ...fields } } ], pagination_token: "..." }
        Each node contains: like_count, comment_count, taken_at, media_type,
        caption: { text: "..." }, view_count, code, pk, id
        """
        client = await self._get_client()

        try:
            resp = await client.post(
                f"{_RAPIDAPI_BASE}/get_ig_user_posts.php",
                headers=self._headers(),
                data={
                    "username_or_url": handle,
                    "amount": str(min(count, 50)),
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Response: { posts: [ { node: { ... } } ] }
            raw_posts = data.get("posts", [])

            posts: list[InstagramPostData] = []
            for item in raw_posts[:count]:
                # Each item is { node: { ... } }
                node = item.get("node", item) if isinstance(item, dict) else {}
                if not isinstance(node, dict):
                    continue

                post = InstagramPostData(
                    post_id=str(node.get("id", node.get("pk", ""))),
                    caption=_extract_caption(node),
                    likes=_safe_int(node.get("like_count", 0)) or 0,
                    comments=_safe_int(node.get("comment_count", 0)) or 0,
                    views=_safe_int(node.get("view_count")),
                    media_type=_classify_media_type(node),
                    timestamp=_safe_int(node.get("taken_at")),
                )
                posts.append(post)

            logger.info(f"IG @{handle}: fetched {len(posts)} recent posts")
            return posts

        except httpx.HTTPStatusError as e:
            logger.warning(f"IG @{handle}: posts fetch error {e.response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"IG @{handle}: posts fetch failed: {e}")
            return []

    @staticmethod
    def _compute_engagement(profile: InstagramProfileData) -> None:
        """Compute average engagement metrics from recent posts."""
        if not profile.recent_posts:
            return

        total_likes = sum(p.likes for p in profile.recent_posts)
        total_comments = sum(p.comments for p in profile.recent_posts)
        count = len(profile.recent_posts)

        profile.avg_likes = total_likes / count if count > 0 else None
        profile.avg_comments = total_comments / count if count > 0 else None

        if profile.followers and profile.followers > 0 and count > 0:
            avg_interactions = (total_likes + total_comments) / count
            profile.engagement_rate = avg_interactions / profile.followers * 100


async def batch_enrich_instagram(
    handles: list[str],
    rapidapi_key: str,
    concurrency: int = 3,
    post_count: int = 12,
    on_progress: callable | None = None,
) -> dict[str, InstagramProfileData | None]:
    """Enrich multiple Instagram profiles concurrently.

    Args:
        handles: List of IG handles to enrich.
        rapidapi_key: RapidAPI key.
        concurrency: Max concurrent API calls (keep low to respect rate limits).
        post_count: Number of recent posts to fetch per creator.
        on_progress: Optional callback(completed, total, handle).

    Returns:
        Dict mapping handle -> InstagramProfileData (or None if failed).
    """
    enricher = InstagramEnricher(rapidapi_key)
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, InstagramProfileData | None] = {}
    completed = 0
    total = len(handles)

    async def _enrich_one(handle: str) -> None:
        nonlocal completed
        async with sem:
            result = await enricher.enrich_profile(handle, post_count=post_count)
            results[handle] = result
            completed += 1

            if on_progress:
                try:
                    await on_progress(completed, total, handle)
                except Exception:
                    pass

    tasks = [_enrich_one(h) for h in handles]
    await asyncio.gather(*tasks)

    await enricher.close()

    success = sum(1 for v in results.values() if v is not None)
    logger.info(f"IG enrichment complete: {success}/{total} profiles fetched")
    return results


# --- Utility ---

def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _extract_caption(node: dict) -> str:
    """Extract caption text from post node.

    Stable API format: caption is an object { text: "...", created_at: ..., pk: ... }
    """
    caption = node.get("caption")
    if isinstance(caption, dict):
        return caption.get("text", "")
    if isinstance(caption, str):
        return caption
    return ""


def _classify_media_type(node: dict) -> str:
    """Classify post media type from the Stable API response.

    media_type values from Instagram:
      1 = image
      2 = video
      8 = carousel/album

    Also checks product_type for further classification.
    """
    media_type = node.get("media_type")
    if media_type == 1:
        return "image"
    if media_type == 2:
        return "video"
    if media_type == 8:
        return "carousel"

    # Fallback: check product_type
    product_type = node.get("product_type", "")
    if "carousel" in str(product_type).lower():
        return "carousel"
    if "clip" in str(product_type).lower() or "reel" in str(product_type).lower():
        return "video"

    # Check for video-specific fields
    if node.get("video_duration") or node.get("video_versions"):
        return "video"

    return "image"
