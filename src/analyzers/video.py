from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.models import VideoQualityResult

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Tier 2: Video quality analysis using pyiqa (TOPIQ-NR) + PySceneDetect.

    TOPIQ-NR is a no-reference image quality metric that works on individual
    frames. We sample frames uniformly from the video, score each, and average.
    The score (0-1, higher=better) is used for both aesthetic and technical
    quality dimensions since they are highly correlated in user-generated content.
    """

    def __init__(self, num_sample_frames: int = 8):
        self._quality_model = None
        self._device = None
        self.num_sample_frames = num_sample_frames

    def _load_model(self) -> None:
        """Lazy-load TOPIQ-NR model via pyiqa."""
        if self._quality_model is not None:
            return

        try:
            import pyiqa

            self._device = torch.device("cpu")
            logger.info("Loading TOPIQ-NR model via pyiqa...")
            self._quality_model = pyiqa.create_metric(
                "topiq_nr", device=self._device
            )
            logger.info("TOPIQ-NR model loaded")
        except Exception as e:
            logger.error(f"Failed to load TOPIQ-NR model: {e}")
            raise

    def _extract_frames(self, video_path: Path) -> list[np.ndarray]:
        """Extract uniformly-spaced frames from video using decord.

        Returns list of numpy arrays in (H, W, C) uint8 format.
        """
        try:
            import decord

            decord.bridge.set_bridge("native")
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)

            if total_frames == 0:
                return []

            n = min(self.num_sample_frames, total_frames)
            # Uniform spacing, avoiding first/last few frames (intros/outros)
            margin = min(total_frames // 10, 5)
            usable = total_frames - 2 * margin
            if usable <= 0:
                indices = list(range(min(n, total_frames)))
            else:
                indices = [
                    margin + int(i * usable / n)
                    for i in range(n)
                ]

            frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
            return [frames[i] for i in range(frames.shape[0])]

        except Exception as e:
            logger.warning(f"Frame extraction failed for {video_path}: {e}")
            return []

    def compute_quality_scores(
        self, video_path: Path
    ) -> tuple[float, float] | None:
        """
        Run TOPIQ-NR on sampled frames to get quality scores.

        Returns (aesthetic, technical) in 0-1 range, or None on failure.
        Both values use the same TOPIQ-NR score since this single metric
        captures overall perceptual quality (aesthetic + technical combined).
        """
        self._load_model()
        if self._quality_model is None:
            return None

        frames = self._extract_frames(video_path)
        if not frames:
            return None

        try:
            scores = []
            for frame_np in frames:
                # Convert (H, W, C) uint8 -> (1, C, H, W) float32 [0, 1]
                img_tensor = (
                    torch.from_numpy(frame_np)
                    .permute(2, 0, 1)
                    .float()
                    .div_(255.0)
                    .unsqueeze(0)
                    .to(self._device)
                )
                with torch.no_grad():
                    score = self._quality_model(img_tensor)
                scores.append(float(score.item()))

            if not scores:
                return None

            avg_score = sum(scores) / len(scores)
            # Clamp to 0-1 range
            avg_score = max(0.0, min(1.0, avg_score))

            # Use the same score for both aesthetic and technical
            # (TOPIQ-NR captures both aspects of quality)
            return (avg_score, avg_score)

        except Exception as e:
            logger.warning(f"Quality scoring failed for {video_path}: {e}")
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
        # Step 1: Quality scores via TOPIQ-NR
        quality_scores = self.compute_quality_scores(video_path)
        if quality_scores is None:
            return VideoQualityResult(
                passed=False,
                rejection_reason="Failed to analyze video quality",
            )

        aesthetic, technical = quality_scores
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
