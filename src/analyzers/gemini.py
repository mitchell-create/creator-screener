from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx

from src.models import GeminiResult

logger = logging.getLogger(__name__)

QUALITY_PROMPT = """You are a video production quality analyst evaluating TikTok creator content.
Based on these key frames from a TikTok video, rate the following on a scale of 1-10:

1. lighting: Is the lighting adequate/professional? (1=dark/blown out, 10=well-lit)
2. composition: Are shots well-framed? (1=random/shaky, 10=intentional framing)
3. production_value: Does this look intentionally produced? (1=lazy phone footage, 10=professional)
4. brand_safety: Is the content clean and brand-appropriate? (1=risky, 10=brand-safe)
5. overall: Overall production quality score (1=very poor, 10=excellent)

Respond ONLY with a JSON object containing these five numeric scores. No other text.
Example: {"lighting": 7, "composition": 6, "production_value": 5, "brand_safety": 8, "overall": 6}"""


class BudgetTracker:
    """Tracks Gemini API spend to stay within weekly budget."""

    def __init__(self, weekly_budget_usd: float = 1.0):
        self.weekly_budget = weekly_budget_usd
        self.spent_usd = 0.0
        self._week_start = time.time()

    def can_spend(self, estimated_cost: float = 0.001) -> bool:
        """Check if we can afford another API call."""
        self._maybe_reset_week()
        return (self.spent_usd + estimated_cost) <= self.weekly_budget

    def record_spend(self, cost_usd: float) -> None:
        self._maybe_reset_week()
        self.spent_usd += cost_usd

    def _maybe_reset_week(self) -> None:
        if time.time() - self._week_start > 7 * 24 * 3600:
            self.spent_usd = 0.0
            self._week_start = time.time()


class GeminiAnalyzer:
    """Tier 3: Video quality assessment via Gemini Flash on OpenRouter."""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Estimated cost per call (4 frames + small output)
    EST_COST_PER_CALL = 0.00025

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.0-flash-001",
        weekly_budget_usd: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.budget = BudgetTracker(weekly_budget_usd)

    async def analyze(
        self,
        frame_paths: list[Path],
    ) -> GeminiResult | None:
        """
        Send key frames to Gemini for quality assessment.

        Returns GeminiResult or None if budget exhausted or API fails.
        """
        if not self.api_key:
            logger.warning("OpenRouter API key not configured, skipping Tier 3")
            return None

        if not self.budget.can_spend(self.EST_COST_PER_CALL):
            logger.info("Weekly Gemini budget exhausted, skipping Tier 3")
            return None

        if not frame_paths:
            return None

        # Build multimodal message with frames
        content = [{"type": "text", "text": QUALITY_PROMPT}]
        for fp in frame_paths:
            b64 = _encode_image(fp)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                })

        if len(content) < 2:
            logger.warning("No valid frames to send to Gemini")
            return None

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    self.OPENROUTER_URL,
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            self.budget.record_spend(self.EST_COST_PER_CALL)

            # Parse response
            text = data["choices"][0]["message"]["content"].strip()
            return _parse_gemini_response(text)

        except httpx.HTTPStatusError as e:
            logger.warning(f"Gemini API error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return None


def _encode_image(path: Path) -> str | None:
    """Read and base64-encode an image file."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.debug(f"Failed to encode {path}: {e}")
        return None


def _parse_gemini_response(text: str) -> GeminiResult | None:
    """Parse Gemini's JSON response into a GeminiResult."""
    try:
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        data = json.loads(text.strip())
        return GeminiResult(
            lighting=_clamp(data.get("lighting")),
            composition=_clamp(data.get("composition")),
            production_value=_clamp(data.get("production_value")),
            brand_safety=_clamp(data.get("brand_safety")),
            overall=_clamp(data.get("overall")),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse Gemini response: {e}. Raw: {text[:200]}")
        return None


def _clamp(val, lo: float = 1.0, hi: float = 10.0) -> float | None:
    if val is None:
        return None
    return max(lo, min(hi, float(val)))
