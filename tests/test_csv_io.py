from pathlib import Path

import pytest

from src.io.csv_reader import read_input_csv
from src.io.csv_writer import write_output_csv, write_passed_only_csv
from src.models import AffiliateResult


def test_read_sample_csv(sample_csv_path):
    affiliates = read_input_csv(sample_csv_path)
    assert len(affiliates) == 3
    assert affiliates[0].profile_url == "https://www.tiktok.com/@testuser1"
    assert affiliates[0].engagement_rate == 3.5
    assert affiliates[0].followers == 15000


def test_read_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_input_csv("/nonexistent/path.csv")


def test_read_csv_with_alternate_column_names(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("url,engagement,follower_count\nhttps://tiktok.com/@test,2.5,1000\n")

    affiliates = read_input_csv(csv_path)
    assert len(affiliates) == 1
    assert affiliates[0].profile_url == "https://tiktok.com/@test"
    assert affiliates[0].engagement_rate == 2.5
    assert affiliates[0].followers == 1000


def test_read_csv_missing_url_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("name,followers\ntest,1000\n")

    with pytest.raises(ValueError, match="profile URL column"):
        read_input_csv(csv_path)


def test_write_output_csv(tmp_output):
    results = [
        AffiliateResult(
            profile_url="https://tiktok.com/@good",
            engagement_rate=5.0,
            followers=10000,
            videos_analyzed=5,
            overall_score=0.75,
            passed=True,
        ),
        AffiliateResult(
            profile_url="https://tiktok.com/@bad",
            engagement_rate=1.0,
            followers=500,
            videos_analyzed=5,
            overall_score=0.2,
            passed=False,
            rejection_reason="Poor audio",
            rejection_tier=1,
        ),
    ]

    path = write_output_csv(results, tmp_output)
    assert path.exists()

    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) == 2
    assert bool(df.iloc[0]["passed"]) is True
    assert bool(df.iloc[1]["passed"]) is False


def test_write_passed_only_csv(tmp_output):
    results = [
        AffiliateResult(profile_url="good", passed=True, overall_score=0.8),
        AffiliateResult(profile_url="bad", passed=False, overall_score=0.2),
    ]

    path = write_passed_only_csv(results, tmp_output)
    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["profile_url"] == "good"
