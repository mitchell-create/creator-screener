from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models import AudioResult

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Tier 1: Audio quality analysis using Silero VAD + DNSMOS."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._vad_model = None
        self._vad_utils = None
        self._dnsmos = None

    def _load_vad(self) -> None:
        """Lazy-load Silero VAD model."""
        if self._vad_model is not None:
            return
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._vad_model = model
        self._vad_utils = utils
        logger.info("Silero VAD loaded")

    def _load_dnsmos(self) -> None:
        """Lazy-load DNSMOS model."""
        if self._dnsmos is not None:
            return
        try:
            from speechmos import dnsmos
            self._dnsmos = dnsmos
            logger.info("DNSMOS loaded via speechmos")
        except ImportError:
            logger.warning("speechmos not available, DNSMOS scoring disabled")

    def extract_audio(self, video_path: Path) -> Path | None:
        """Extract audio from video to 16kHz mono WAV using ffmpeg."""
        wav_path = video_path.with_suffix(".wav")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-vn",
                "-f", "wav",
                str(wav_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"ffmpeg failed: {result.stderr[:200]}")
                return None
            return wav_path if wav_path.exists() else None
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None

    def compute_speech_percentage(self, audio: np.ndarray) -> float:
        """Run Silero VAD to detect what percentage of audio contains speech."""
        self._load_vad()
        if self._vad_model is None:
            return 0.0

        get_speech_timestamps = self._vad_utils[0]

        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / 32768.0

        try:
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=self.sample_rate,
                threshold=0.5,
            )
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            return 0.0

        if not speech_timestamps:
            return 0.0

        total_samples = len(audio_tensor)
        speech_samples = sum(
            ts["end"] - ts["start"] for ts in speech_timestamps
        )
        return (speech_samples / total_samples) * 100.0

    def compute_dnsmos(self, audio: np.ndarray) -> dict[str, float] | None:
        """Run DNSMOS to get speech quality scores."""
        self._load_dnsmos()
        if self._dnsmos is None:
            return None

        try:
            # speechmos expects float32 array
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            result = self._dnsmos.run(audio, sr=self.sample_rate)
            return {
                "ovrl": result.get("ovrl_mos", result.get("OVRL", 0.0)),
                "sig": result.get("sig_mos", result.get("SIG", 0.0)),
                "bak": result.get("bak_mos", result.get("BAK", 0.0)),
            }
        except Exception as e:
            logger.warning(f"DNSMOS failed: {e}")
            return None

    def analyze(
        self,
        video_path: Path,
        speech_min_pct: float = 10.0,
        dnsmos_ovrl_min: float = 2.5,
    ) -> AudioResult:
        """Full audio analysis pipeline for a single video."""
        # Step 1: Extract audio
        wav_path = self.extract_audio(video_path)
        if wav_path is None:
            return AudioResult(
                passed=False,
                rejection_reason="Failed to extract audio",
            )

        try:
            # Load audio
            import torchaudio
            waveform, sr = torchaudio.load(str(wav_path))
            audio_np = waveform.squeeze().numpy()

            # Step 2: VAD - check if there's speech
            speech_pct = self.compute_speech_percentage(audio_np)

            if speech_pct < speech_min_pct:
                return AudioResult(
                    speech_pct=speech_pct,
                    passed=False,
                    rejection_reason=f"Low speech detected ({speech_pct:.1f}% < {speech_min_pct}%)",
                )

            # Step 3: DNSMOS quality scoring
            dnsmos_scores = self.compute_dnsmos(audio_np)
            if dnsmos_scores is None:
                # Can't score, but speech exists - pass with warning
                return AudioResult(
                    speech_pct=speech_pct,
                    passed=True,
                )

            ovrl = dnsmos_scores["ovrl"]
            passed = ovrl >= dnsmos_ovrl_min
            rejection = None if passed else f"Poor audio quality (DNSMOS {ovrl:.2f} < {dnsmos_ovrl_min})"

            return AudioResult(
                speech_pct=speech_pct,
                dnsmos_ovrl=ovrl,
                dnsmos_sig=dnsmos_scores["sig"],
                dnsmos_bak=dnsmos_scores["bak"],
                passed=passed,
                rejection_reason=rejection,
            )

        except Exception as e:
            logger.warning(f"Audio analysis failed for {video_path}: {e}")
            return AudioResult(
                passed=False,
                rejection_reason=f"Audio analysis error: {str(e)[:100]}",
            )
        finally:
            # Clean up WAV file
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)
