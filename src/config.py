from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_TEMP_DIR = str(Path(tempfile.gettempdir()) / "affiliate_pipeline")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AFF_", env_file=".env", extra="ignore")

    # --- Acquisition ---
    videos_per_profile: int = 5
    download_concurrency: int = 10
    download_timeout_seconds: int = 120
    temp_dir: str = _DEFAULT_TEMP_DIR
    download_max_resolution: int = 720

    # --- Tier 1: Audio ---
    vad_speech_min_pct: float = 10.0
    dnsmos_ovrl_min: float = 2.5
    dnsmos_sig_min: float = 2.0
    audio_sample_rate: int = 16000

    # --- Tier 2: Video ---
    dover_aesthetic_min: float = 0.4
    dover_technical_min: float = 0.4
    dover_overall_min: float = 0.45
    min_scene_cuts: int = 2

    # --- Tier 3: Gemini ---
    tier3_enabled: bool = True
    borderline_band_pct: float = 15.0
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.0-flash-001"
    gemini_max_frames: int = 4
    gemini_min_score: float = 6.0
    gemini_weekly_budget_usd: float = 1.00

    # --- Pipeline ---
    batch_size: int = 15
    max_concurrent_analysis: int = 10
    per_affiliate_timeout: int = 180  # 3 minutes max per affiliate
    dedup_enabled: bool = True

    # --- I/O ---
    input_csv_path: str = "./data/input.csv"
    output_csv_path: str = "./data/output.csv"

    # --- Tier Failure Rate ---
    tier_failure_rate: float = 0.5  # Reject affiliate if more than this fraction of videos fail a tier

    # --- Scoring Weights (Video Quality) ---
    weight_audio: float = 0.35
    weight_video: float = 0.35
    weight_edits: float = 0.15
    weight_engagement: float = 0.15

    # --- Creator Enrichment & Scoring ---
    firecrawl_api_key: str = ""
    rapidapi_key: str = ""  # RapidAPI key for Instagram Scraper API
    enrichment_enabled: bool = True
    enrichment_concurrency: int = 5  # Concurrent scrapes

    # Modash pipeline settings
    modash_tiktok_concurrency: int = 5  # Concurrent yt-dlp TikTok checks
    modash_ig_concurrency: int = 3  # Concurrent RapidAPI IG calls (keep low for rate limits)
    modash_ig_post_count: int = 12  # Recent posts to fetch per IG creator

    # Creator score weights (0-1, auto-normalized)
    creator_weight_engagement: float = 0.35
    creator_weight_follower_quality: float = 0.25
    creator_weight_consistency: float = 0.20
    creator_weight_authenticity: float = 0.20

    # Pre-filter: reject creators below this creator_score before video analysis
    # Set to 0.0 to disable pre-filtering
    creator_score_min: float = 0.0

    # Brand-fit matching
    brand_fit_enabled: bool = False
    embedding_model: str = "openai/text-embedding-3-small"
    use_local_embeddings: bool = False  # Use sentence-transformers instead of API

    # --- Discovery ---
    discovery_max_creators: int = 100
    discovery_platforms: str = "tiktok,instagram"  # Comma-separated

    # --- Slack Bot ---
    slack_bot_token: str = ""    # xoxb-... OAuth bot token
    slack_app_token: str = ""    # xapp-... Socket Mode app-level token
    slack_channel_id: str = ""   # Channel ID to watch for CSV uploads

    # --- Supabase ---
    supabase_url: str = ""       # https://xxxx.supabase.co
    supabase_key: str = ""       # service_role or anon key

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir)
