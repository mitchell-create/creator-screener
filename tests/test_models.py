from src.models import (
    AffiliateInput,
    AffiliateResult,
    AudioResult,
    GeminiResult,
    PipelineStats,
    VideoQualityResult,
    VideoScore,
)


def test_affiliate_input():
    a = AffiliateInput(profile_url="https://tiktok.com/@test")
    assert a.engagement_rate is None
    assert a.followers is None


def test_affiliate_input_full():
    a = AffiliateInput(
        profile_url="https://tiktok.com/@test",
        engagement_rate=3.5,
        followers=10000,
    )
    assert a.engagement_rate == 3.5


def test_audio_result_defaults():
    r = AudioResult()
    assert r.speech_pct == 0.0
    assert r.passed is False


def test_video_score_nesting():
    vs = VideoScore(
        video_id="123",
        audio=AudioResult(speech_pct=50.0, dnsmos_ovrl=3.5, passed=True),
        video_quality=VideoQualityResult(
            dover_aesthetic=0.6, dover_technical=0.7, scene_cuts=5, passed=True
        ),
    )
    assert vs.audio.dnsmos_ovrl == 3.5
    assert vs.video_quality.scene_cuts == 5


def test_affiliate_result_defaults():
    r = AffiliateResult(profile_url="test")
    assert r.passed is False
    assert r.video_scores == []
    assert r.error is None


def test_pipeline_stats():
    s = PipelineStats()
    assert s.total_input == 0
    assert s.gemini_cost_usd == 0.0
