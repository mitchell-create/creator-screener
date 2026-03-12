import os

from src.config import Settings


def test_default_settings():
    s = Settings()
    assert s.videos_per_profile == 5
    assert s.vad_speech_min_pct == 10.0
    assert s.dnsmos_ovrl_min == 2.5
    assert s.dover_overall_min == 0.45
    assert s.batch_size == 50
    assert s.tier3_enabled is True


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("AFF_VIDEOS_PER_PROFILE", "3")
    monkeypatch.setenv("AFF_BATCH_SIZE", "25")
    monkeypatch.setenv("AFF_TIER3_ENABLED", "false")

    s = Settings()
    assert s.videos_per_profile == 3
    assert s.batch_size == 25
    assert s.tier3_enabled is False


def test_temp_path():
    s = Settings()
    assert s.temp_path.name == "affiliate_pipeline"
