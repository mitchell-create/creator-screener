from __future__ import annotations

import logging
from pathlib import Path

from src.models import VideoQualityResult

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Tier 2: Video quality analysis using pyiqa (DOVER) + PySceneDetect."""

    def __init__(self):
        self._dover_model = None
        self._device = None

    def _load_dover(self) -> None:
        """Lazy-load DOVER model via pyiqa."""
        if self._dover_model is not None:
            return

        try:
            import pyiqa
            import torch

            self._device = torch.device("cpu")
            logger.info("Loading DOVER model via pyiqa...")
            self._dover_model = pyiqa.create_metric("dover", device=self._device)
            logger.info("DOVER model loaded")
        except Exception as e:
            logger.error(f"Failed to load DOVER model: {e}")
            raise

    def compute_dover_scores(self, video_path: Path) -> tuple[float, float] | None:
        """
        Run DOVER to get aesthetic and technical quality scores.

        Returns (aesthetic, technical) normalized to 0-1 range, or None on failure.
        """
        self._load_dover()
        if self._dover_model is None:
            return None

        try:
            score = self._dover_model(str(video_path))

            # pyiqa's DOVER returns a dict or tensor depending on version
            if isinstance(score, dict):
                aesthetic = float(score.get("aesthetic", score.get("a", 0)))
                technical = float(score.get("technical", score.get("t", 0)))
            elif hasattr(score, "item"):
                # Single score - use as both
                val = float(score.item()) if hasattr(score, "item") else float(score)
                aesthetic = val
                technical = val
            else:
                aesthetic = float(score)
                technical = float(score)

            # Normalize to 0-1 if needed (DOVER raw scores can vary)
            aesthetic = max(0.0, min(1.0, aesthetic))
            technical = max(0.0, min(1.0, technical))

            return (aesthetic, technical)

        except Exception as e:
            logger.warning(f"DOVER scoring failed for {video_path}: {e}")
            return None

    def count_scene_cuts(self, video_path: Path) -> int:
        """Count the number of scene cuts/transitions using PySceneDetect."""
        try:
            from scenedetect import detect, ContentDetector

            scenes = detect(str(video_path), ContentDetector(threshold=27.0))
            cut_count = max(0, len(scenes) - 1)
            logger.debug(f"Detected {cut_count} cuts in {video_path.name}")
            return cut_count

        except Exception as e:
            logger.warning(f"Scene detection failed for {video_path}: {e}")
            return 0

    def analyze(
        self,
        video_path: Path,
        aesthetic_min: float = 0.4,
        technical_min: float = 0.4,
        overall_min: float = 0.45,
        min_cuts: int = 2,
    ) -> VideoQualityResult:
        """Full video quality analysis for a single video."""
        # Step 1: DOVER quality scores
        dover_scores = self.compute_dover_scores(video_path)
        if dover_scores is None:
            return VideoQualityResult(
                passed=False,
                rejection_reason="Failed to analyze video quality",
            )

        aesthetic, technical = dover_scores
        overall = (aesthetic + technical) / 2.0

        # Step 2: Scene cut detection
        cuts = self.count_scene_cuts(video_path)

        # Step 3: Pass/fail evaluation
        reasons = []
        if aesthetic < aesthetic_min:
            reasons.append(f"Low aesthetic ({aesthetic:.2f} < {aesthetic_min})")
        if technical < technical_min:
            reasons.append(f"Low technical ({technical:.2f} < {technical_min})")
        if overall < overall_min:
            reasons.append(f"Low overall ({overall:.2f} < {overall_min})")

        passed = len(reasons) == 0
        rejection = "; ".join(reasons) if reasons else None

        return VideoQualityResult(
            dover_aesthetic=aesthetic,
            dover_technical=technical,
            scene_cuts=cuts,
            passed=passed,
            rejection_reason=rejection,
        )
