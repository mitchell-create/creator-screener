"""Generate personalized outreach emails and DMs for affiliate creators.

For each creator, uses Gemini to generate:
  1. A personalized opener referencing something specific about their content
  2. A personalized closer tying their content to SecondKind's postbiotic product

The middle section (SecondKind pitch + partnership terms) is templated and
consistent across all creators.

Input: scored CSV from the modash pipeline (needs content_fit_passed creators)
Output: CSV with full email body + DM body per creator
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
import pandas as pd

from src.acquisition.tiktok_enricher import enrich_tiktok_profile
from src.acquisition.instagram_enricher import InstagramEnricher
from src.config import Settings

logger = logging.getLogger("outreach")


# ---------------------------------------------------------------------------
# Brand-specific copy — edit this for different campaigns
# ---------------------------------------------------------------------------

BRAND_NAME = "SecondKind"
BRAND_URL = "https://secondkind.com"

# Hook + explainer — sits after the partnership/closer paragraph, right before
# the CTA. The hook differentiates postbiotics vs probiotics in one beat; the
# explainer answers it.
HOOK_LINE = "Have you heard of postbiotics, not probiotics?"

EMAIL_BRAND_PITCH = (
    "Postbiotics are the finished compound your body actually uses — "
    "essentially the beneficial compound probiotics try to deliver, but most "
    "don't survive digestion long enough to do it."
)

DM_BRAND_PITCH = EMAIL_BRAND_PITCH

# Partnership pitch — starts with "That's exactly why we're reaching out" so it
# flows naturally from the interpretive second sentence of the opener.
EMAIL_PARTNERSHIP = (
    "That's exactly why we're reaching out. We're looking for creators who "
    "actually resonate with their audience, not ones pushing every big "
    "supplement brand you've seen a hundred times."
)

DM_PARTNERSHIP = (
    "That's exactly why we're reaching out — we're looking for creators who "
    "actually resonate with their audience, not ones pushing every big "
    "supplement brand you've seen a hundred times."
)

EMAIL_CTA = (
    "Open to a quick call to talk through it?\n\n"
    "Remy\n"
    "Founder, SecondKind"
)
DM_CTA = "Open to a quick call?"


# ---------------------------------------------------------------------------
# Personalization via Gemini
# ---------------------------------------------------------------------------

async def generate_personalization(
    handle: str,
    name: str,
    bio: str,
    content_samples: list[str],
    openrouter_api_key: str,
    openrouter_model: str = "google/gemini-2.0-flash-001",
) -> dict:
    """Generate personalized opener and closer for one creator via Gemini.

    Returns dict with 'opener' and 'closer' strings.
    """
    content_sample = "\n---\n".join(content_samples[:10])
    if len(content_sample) > 2500:
        content_sample = content_sample[:2500] + "...[truncated]"

    prompt = f"""You're writing cold outreach for SecondKind (postbiotic supplement). You are reaching out to a creator about a long-term partnership.

The tone you're going for: matter-of-fact. Like a real person who actually watched their content wrote this in 30 seconds. Not an influencer marketing agent. Not someone trying to flatter them. Just observational.

**Creator:** @{handle}  |  **Name:** {name or "(unknown)"}
**Bio:** {bio or "(none)"}

**Their content:**
{content_sample}

Posts tagged [RECENT POST] are from the last 10 days. ONLY reference specific posts if they are tagged [RECENT POST]. For older content, you can reference general themes but do NOT reference specific posts.

Write TWO things:

1. **OPENER** (2-3 sentences, ~40-60 words):
   TWO parts that flow together:

   a) A specific OBSERVATION about their content (what you noticed — a post, phrase, theme).
   b) A brief INTERPRETATION of what that observation means about them — their authenticity, why their audience trusts them, why it's unusual for the space.

   DO NOT end with "That's exactly why we're reaching out" or any similar phrase — we add that static transition after your opener automatically. Just end your opener at the interpretation sentence.

   GOOD examples (observation → interpretation bridge):
   - "I noticed you mentioned period cramps affecting your running, which a lot of fitness content tends to gloss over. It's the kind of vulnerability most creators edit out, and it's probably why your audience actually trusts what you say."
   - "Saw your post about GLP-1 and the 'losing people at your worst' line. Most weight loss content stops at the before/afters. Being that honest about the emotional side is why people keep coming back to your content."
   - "Your 3-2-8 method post came up in my feed. The 'working with the body not against it' framing is the part I haven't seen from other PCOS creators. Feels like something you actually live by, not just another framework."

   BAD examples (cheesy, trying too hard, rating them):
   - "Okay, the 'one wipe guarantee' is gold."
   - "That post really hit."
   - "is really smart" / "is refreshing" / "is so real"
   - "got me" / "cracked me up"
   - "You're empowering women" / "You get that..."
   - Any "you're doing amazing" / "you're amazing" energy

   The interpretive sentence should feel like a real observation — not flattery. Think: "and that's why your audience trusts you" or "which is not a common angle in this space" or "which is probably why people come to you for advice instead of generic wellness accounts."

2. **CLOSER** (1 sentence, ~10-15 words):
   One line connecting their content to why SecondKind specifically makes sense for them. No "natural fit" / "could be a useful resource" / "might resonate" language. Just state the connection plainly.

   GOOD examples:
   - "A lot of your audience is probably dealing with gut stuff from the meds."
   - "Figured the gut-skin angle overlaps with what you're already covering."
   - "Seemed like your audience would care about how this actually works."

   BAD examples:
   - "SecondKind could be a powerful addition to their toolkit"
   - "This could be a natural fit"
   - "might resonate"

STRICT RULES (break these and the message sounds AI-written):
- No "Okay, [X] is gold/amazing/refreshing"
- No rating words: gold, smart, real, genuine, refreshing, powerful, rare, brilliant
- No "hit hard" / "got me" / "cracked me up" / "really stuck with me"
- No em dashes (use commas or periods)
- No "not just X, but Y" constructions
- No "It's clear that..." / "You can tell..."
- No "resonates" / "aligns" / "empowers"
- No exclamation points
- Write like a text, not a marketing email
- If you have no recent specific posts to reference, stay general about themes

Respond with ONLY a JSON object:
{{"opener": "<1-2 sentences>", "closer": "<1 sentence>"}}"""

    # Retry up to 5 times on transient errors with exponential backoff
    last_err = None
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": openrouter_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 400,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]

                parsed = json.loads(content)
                return {
                    "opener": parsed.get("opener", "").strip(),
                    "closer": parsed.get("closer", "").strip(),
                }

        except Exception as e:
            last_err = e
            if attempt < 4:
                # Exponential backoff: 5s, 10s, 20s, 40s
                delay = 5 * (2 ** attempt)
                logger.info(f"@{handle}: retry {attempt + 1}/5 after {delay}s ({e.__class__.__name__})")
                await asyncio.sleep(delay)
                continue

    logger.warning(f"Personalization failed for @{handle} after retries: {last_err}")
    return {"opener": "", "closer": ""}


# ---------------------------------------------------------------------------
# Assemble messages
# ---------------------------------------------------------------------------

def _clean_opener(opener: str) -> str:
    """Strip duplicate greetings and trailing pivot phrases from the LLM-generated opener."""
    import re as _re
    # Remove leading "Hey!" "Hi!" "Hey @handle," etc.
    opener = _re.sub(r"^(hey|hi|hello)[!,\s]+[^\s]*[,!\s]+", "", opener, flags=_re.IGNORECASE).strip()

    # Strip trailing pivot phrases that duplicate our static transition.
    # LLM sometimes ends with "That's exactly why we're reaching out..." or similar,
    # which duplicates the static partnership line that follows.
    trailing_patterns = [
        r"\s*that'?s (exactly )?why we'?re reaching out.*$",
        r"\s*that'?s (exactly )?why i'?m reaching out.*$",
        r"\s*which is (exactly )?why we'?re reaching out.*$",
    ]
    for pat in trailing_patterns:
        opener = _re.sub(pat, "", opener, flags=_re.IGNORECASE | _re.DOTALL).strip()

    # Remove trailing ellipsis/periods that were just holding a cut-off phrase
    opener = opener.rstrip(".… ").rstrip() + "."

    # Capitalize first letter if it got lowercased
    if opener and opener[0].islower():
        opener = opener[0].upper() + opener[1:]
    return opener


def build_email(
    name: str,
    opener: str,
    closer: str,
    handle: str = "",
) -> str:
    """Build a concise email.

    Order: opener → partnership framing (with personalized closer) → brand pitch → CTA.
    This acknowledges THEM before explaining the product, which reads less like a
    sales pitch and more like targeted outreach.
    """
    first_name = _first_name(name, handle)
    greeting = f"Hi {first_name}," if first_name else "Hey,"
    opener = _clean_opener(opener)

    return f"""{greeting}

{opener}

{EMAIL_PARTNERSHIP} {closer}

{HOOK_LINE} {EMAIL_BRAND_PITCH}

{EMAIL_CTA}"""


def build_dm(
    name: str,
    opener: str,
    closer: str,
    handle: str = "",
) -> str:
    """Build a very short DM with the same opener → partnership → brand flow."""
    first_name = _first_name(name, handle)
    greeting = f"Hey {first_name}!" if first_name else "Hey!"
    opener = _clean_opener(opener)

    return f"""{greeting} {opener}

{DM_PARTNERSHIP} {closer}

{HOOK_LINE} {DM_BRAND_PITCH}

{DM_CTA}"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def generate_outreach(
    input_csv: str,
    output_csv: str,
    only_passed: bool = True,
    top_n: int | None = None,
    refresh_content: bool = False,
    concurrency: int = 2,
) -> None:
    """Generate outreach messages for all creators in a scored CSV.

    Args:
        input_csv: Scored CSV from modash pipeline.
        output_csv: Output CSV path.
        only_passed: Only process content_fit_passed creators.
        top_n: Limit to top N creators by affiliate_score.
        refresh_content: Re-fetch TikTok descriptions + IG captions live.
            Otherwise, uses ig_bio + recommendation text as best-effort context.
        concurrency: Max concurrent LLM calls.
    """
    settings = Settings()
    if not settings.openrouter_api_key:
        logger.error("AFF_OPENROUTER_API_KEY required")
        sys.exit(1)

    df = pd.read_csv(input_csv)

    if only_passed and "content_fit_passed" in df.columns:
        df = df[df["content_fit_passed"] == True]

    if "affiliate_score" in df.columns:
        df = df.sort_values("affiliate_score", ascending=False)

    if top_n:
        df = df.head(top_n)

    df = df.reset_index(drop=True)

    logger.info(f"Generating outreach for {len(df)} creators")

    # --- Optionally re-fetch live content for richer personalization ---
    content_by_handle: dict[str, tuple[str, list[str]]] = {}  # handle -> (bio, [content])

    if refresh_content:
        logger.info("Refreshing content from TikTok + Instagram (sequential to avoid rate limits)...")
        ig_enricher = InstagramEnricher(settings.rapidapi_key) if settings.rapidapi_key else None

        # Sequential IG calls with delay to avoid rate limits
        # TikTok can run in parallel since it's yt-dlp (no rate limit)
        import random as _random

        async def _fetch_tt(row) -> tuple[str, list[str]]:
            handle = _clean_handle(row.get("ig_handle") or row.get("ig_profile", ""))
            tt_handle = _clean_tt_handle(row.get("tiktok_handle") or row.get("tiktok_profile", ""))
            tt_descriptions: list[str] = []
            if tt_handle:
                tk = await enrich_tiktok_profile(tt_handle, video_count=10)
                if tk:
                    tt_descriptions = tk.get("descriptions", [])
            return handle, tt_descriptions

        # TikTok in parallel (yt-dlp handles its own pacing)
        tt_results = await asyncio.gather(*[_fetch_tt(row) for _, row in df.iterrows()])
        tt_by_handle = {h: d for h, d in tt_results}

        # Only reference posts from the last 10 days for specific-post references
        import time
        ten_days_ago = int(time.time()) - (10 * 86400)

        # IG sequentially with 2-4 second delay between each
        for idx, (_, row) in enumerate(df.iterrows()):
            handle = _clean_handle(row.get("ig_handle") or row.get("ig_profile", ""))
            ig_bio = ""
            recent_ig_captions: list[str] = []
            older_ig_captions: list[str] = []

            if ig_enricher and handle:
                ig = await ig_enricher.enrich_profile(handle, post_count=12, delay_range=(2.0, 4.0))
                if ig:
                    ig_bio = ig.bio or ""
                    for p in ig.recent_posts:
                        if not p.caption:
                            continue
                        if p.timestamp and p.timestamp >= ten_days_ago:
                            recent_ig_captions.append(f"[RECENT POST] {p.caption}")
                        else:
                            older_ig_captions.append(p.caption)
                logger.info(
                    f"[{idx+1}/{len(df)}] Fetched IG @{handle}: bio={len(ig_bio)} chars, "
                    f"{len(recent_ig_captions)} recent (<=10d) captions, "
                    f"{len(older_ig_captions)} older captions"
                )

            # Content order: recent IG posts first (for specific references), then older context
            content_by_handle[handle] = (
                ig_bio,
                recent_ig_captions + tt_by_handle.get(handle, []) + older_ig_captions,
            )

        if ig_enricher:
            await ig_enricher.close()
    else:
        # Use existing bio + recommendation as context
        for _, row in df.iterrows():
            handle = _clean_handle(row.get("ig_handle") or row.get("ig_profile", ""))
            bio = str(row.get("ig_bio", "")) if pd.notna(row.get("ig_bio")) else ""
            rec = str(row.get("recommendation", "")) if pd.notna(row.get("recommendation")) else ""
            collabs = str(row.get("recent_collabs", "")) if pd.notna(row.get("recent_collabs")) else ""
            # Best-effort context from what we already have
            content_by_handle[handle] = (
                bio,
                [s for s in [bio, rec, collabs] if s and s != "nan"],
            )

    # --- Generate personalized messages (sequential with pacing to avoid rate limits) ---
    results: list[dict] = []
    completed = 0
    total = len(df)

    async def _generate_one(row) -> None:
        nonlocal completed
        handle = _clean_handle(row.get("ig_handle") or row.get("ig_profile", ""))
        tt_handle = _clean_tt_handle(row.get("tiktok_handle") or row.get("tiktok_profile", ""))
        name = str(row.get("name", "")).strip() if pd.notna(row.get("name")) else ""
        email = str(row.get("email", "")).strip() if pd.notna(row.get("email")) else ""

        bio, content = content_by_handle.get(handle, ("", []))

        personalization = await generate_personalization(
            handle=handle,
            name=name,
            bio=bio,
            content_samples=content,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
        )

        opener = personalization["opener"]
        closer = personalization["closer"]

        email_body = build_email(name, opener, closer, handle=handle)
        dm_body = build_dm(name, opener, closer, handle=handle)

        completed += 1
        logger.info(f"[{completed}/{total}] @{handle}: generated")

        results.append({
            "rank": row.get("rank", ""),
            "name": name,
            "ig_handle": f"@{handle}",
            "ig_profile": f"https://instagram.com/{handle}",
            "tiktok_profile": f"https://tiktok.com/@{tt_handle}" if tt_handle else "",
            "email": email,
            "affiliate_score": row.get("affiliate_score", ""),
            "personalized_opener": opener,
            "personalized_closer": closer,
            "email_subject": (
                f"{_first_name(name, handle)}, quick idea" if _first_name(name, handle)
                else "A quick idea"
            ),
            "email_body": email_body,
            "dm_body": dm_body,
        })

    # Run sequentially with 8s pacing between calls to stay under OpenRouter rate limits.
    # Save progress after each successful generation so failures don't lose work.
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, (_, row) in enumerate(df.iterrows()):
        await _generate_one(row)
        # Save incremental progress after every successful call
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)
        if idx < len(df) - 1:
            await asyncio.sleep(8.0)

    # Sort by rank
    results.sort(key=lambda r: float(r.get("rank") or 9999))

    out_df = pd.DataFrame(results)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(results)} outreach messages to {out_path}")


def _first_name(name: str, handle: str = "") -> str:
    """Extract a usable first name, falling back to handle-derived guess."""
    if name and name.strip() and name.strip().lower() != "nan":
        # Strip qualifiers like "| Metro-Detroit Doctor", titles
        first = name.strip().split("|")[0].strip().split(",")[0].strip()
        first = first.split()[0] if first else ""
        # Skip titles
        if first.lower() in ("dr.", "dr", "mr.", "mr", "mrs.", "ms.", "ms"):
            parts = name.strip().split()
            if len(parts) > 1:
                first = parts[1]
        if first and first[0].isalpha():
            return first
    # No name — don't guess from handle (often unreadable)
    return ""


def _clean_handle(val) -> str:
    s = str(val) if val is not None else ""
    s = s.replace("https://instagram.com/", "").replace("https://www.instagram.com/", "")
    return s.strip("@/").strip()


def _clean_tt_handle(val) -> str:
    s = str(val) if val is not None else ""
    s = s.replace("https://tiktok.com/@", "").replace("https://www.tiktok.com/@", "")
    return s.strip("@/").strip()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate personalized outreach messages")
    parser.add_argument("--input", "-i", required=True, help="Scored CSV input path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--top", type=int, default=None, help="Limit to top N by affiliate_score")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch live TikTok/IG content for richer personalization")
    parser.add_argument("--all", action="store_true", help="Include content-fit failures too")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(generate_outreach(
        input_csv=args.input,
        output_csv=args.output,
        only_passed=not args.all,
        top_n=args.top,
        refresh_content=args.refresh,
    ))


if __name__ == "__main__":
    _main()
