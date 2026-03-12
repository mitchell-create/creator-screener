from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path():
    return FIXTURES_DIR / "sample_input.csv"


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output.csv"
