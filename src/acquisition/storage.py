from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class TempVideoStorage:
    """Manages temporary directories for downloaded videos."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_affiliate_dir(self, affiliate_id: str) -> Path:
        """Create a temp directory for an affiliate's videos."""
        safe_id = affiliate_id.replace("/", "_").replace("@", "").replace(":", "_")
        affiliate_dir = self.base_dir / safe_id
        affiliate_dir.mkdir(parents=True, exist_ok=True)
        return affiliate_dir

    def cleanup_affiliate(self, affiliate_id: str) -> None:
        """Remove all temp files for a specific affiliate."""
        safe_id = affiliate_id.replace("/", "_").replace("@", "").replace(":", "_")
        affiliate_dir = self.base_dir / safe_id
        if affiliate_dir.exists():
            shutil.rmtree(affiliate_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp files for {affiliate_id}")

    def cleanup_all(self) -> None:
        """Remove all temp files."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all temp files")

    def get_disk_usage_mb(self) -> float:
        """Get total disk usage of temp directory in MB."""
        if not self.base_dir.exists():
            return 0.0
        total = sum(f.stat().st_size for f in self.base_dir.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    def get_affiliate_id(self, profile_url: str) -> str:
        """Extract a safe identifier from a TikTok profile URL."""
        # https://www.tiktok.com/@username -> username
        url = profile_url.rstrip("/")
        if "/@" in url:
            return url.split("/@")[-1]
        return url.split("/")[-1]
