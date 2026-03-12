from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.scoring.aggregator import AggregateScore


@dataclass
class ThresholdResult:
    passed: bool
    is_borderline: bool
    rejection_reason: str | None = None


def evaluate_affiliate(
    aggregate: AggregateScore,
    settings: Settings,
) -> ThresholdResult:
    """Apply pass/fail logic and detect borderline cases for Tier 3."""
    if aggregate.overall is None:
        return ThresholdResult(
            passed=False,
            is_borderline=False,
            rejection_reason="Insufficient data to score",
        )

    # Normalized pass threshold (settings use 0-1 scale)
    pass_threshold = settings.dover_overall_min

    if aggregate.overall < pass_threshold:
        return ThresholdResult(
            passed=False,
            is_borderline=False,
            rejection_reason=f"Overall score {aggregate.overall:.3f} below threshold {pass_threshold}",
        )

    # Check if borderline (within X% above threshold)
    borderline_ceiling = pass_threshold * (1.0 + settings.borderline_band_pct / 100.0)
    is_borderline = aggregate.overall < borderline_ceiling

    return ThresholdResult(
        passed=True,
        is_borderline=is_borderline,
    )
