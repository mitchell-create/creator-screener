from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AFF_", env_file=".env", extra="ignore")

    # --- Acquisition ---
    videos_per_profile: int = 5
    download_concurrency: int = 10
    download_timeout_seconds: int = 120
    temp_dir: str = "/tmp/affiliate_pipeline"
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
    batch_size: int = 50
    max_concurrent_analysis: int = 4

    # --- I/O ---
    input_csv_path: str = "./data/input.csv"
    output_csv_path: str = "./data/output.csv"

    # --- Tier Failure Rate ---
    tier_failure_rate: float = 0.5  # Reject affiliate if more than this fraction of videos fail a tier

    # --- Scoring Weights ---
    weight_audio: float = 0.35
    weight_video: float = 0.35
    weight_edits: float = 0.15
    weight_engagement: float = 0.15

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
