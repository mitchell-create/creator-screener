# TikTok Affiliate Video Quality Analyzer

Automated screening of TikTok affiliate creator production quality. Given a CSV of affiliate profiles, downloads their recent videos and analyzes audio/video quality through a tiered pipeline to filter out low-quality creators.

## How It Works

```
CSV (affiliates)
  |
  v
[yt-dlp] Download 5 most recent videos per profile
  |
  v
[TIER 1 - FREE] Audio Analysis (Silero VAD + DNSMOS)
  - Speech < 10%? REJECT
  - DNSMOS overall < 2.5? REJECT
  |
  v
[TIER 2 - FREE] Video Quality (TOPIQ-NR + PySceneDetect)
  - Quality score below threshold? REJECT
  |
  v
[TIER 3 - CHEAP] Gemini Flash via OpenRouter (borderline only)
  - Key frames sent for structured scoring
  - Only ~10-20% of affiliates reach here
  |
  v
Output: Filtered CSV with scores + pass/fail + rejection reasons
```

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (must be in PATH)

### Setup

```bash
# Clone and enter directory
git clone https://github.com/mitchell-create/AffiliatePipeline.git
cd AffiliatePipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Input CSV Format

Create a CSV at `data/input.csv` with at minimum a `profile_url` column:

```csv
profile_url,engagement_rate,followers
https://www.tiktok.com/@username1,5.2,50000
https://www.tiktok.com/@username2,3.8,120000
```

Accepted column names: `profile_url`/`url`/`tiktok_url`/`link`, `engagement_rate`/`er`, `followers`/`follower_count`.

### Run

```bash
# Full run
python -m src.main --input data/input.csv --output data/output.csv

# Limit to first N affiliates
python -m src.main -i data/input.csv -o data/output.csv --limit 10

# Dry run (parse CSV without downloading/analyzing)
python -m src.main -i data/input.csv --dry-run

# Verbose logging
python -m src.main -i data/input.csv -o data/output.csv --verbose
```

### Output

Two CSV files are generated:
- `output.csv` - All affiliates with scores, pass/fail, and rejection reasons
- `output_passed.csv` - Only affiliates that passed all tiers

Output columns: `profile_url`, `engagement_rate`, `followers`, `videos_analyzed`, `avg_audio_score`, `avg_video_aesthetic`, `avg_video_technical`, `avg_scene_cuts`, `gemini_avg_score`, `overall_score`, `passed`, `rejection_reason`, `rejection_tier`, `error`

## Configuration

All settings are configured via environment variables (prefix `AFF_`). See `.env.example` for the full list.

### Key Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AFF_VAD_SPEECH_MIN_PCT` | 10.0 | Minimum speech percentage to pass Tier 1 |
| `AFF_DNSMOS_OVRL_MIN` | 2.5 | Minimum DNSMOS overall score (1-5 scale) |
| `AFF_DOVER_OVERALL_MIN` | 0.45 | Minimum video quality score (0-1 scale) |
| `AFF_DOVER_AESTHETIC_MIN` | 0.4 | Minimum aesthetic score (0-1 scale) |
| `AFF_DOVER_TECHNICAL_MIN` | 0.4 | Minimum technical score (0-1 scale) |
| `AFF_MIN_SCENE_CUTS` | 2 | Minimum scene cuts per video |
| `AFF_TIER_FAILURE_RATE` | 0.5 | Fraction of videos that must fail to reject an affiliate |
| `AFF_GEMINI_MIN_SCORE` | 6.0 | Minimum Gemini overall score (1-10 scale) |

### Scoring Weights

The overall affiliate score is a weighted composite:

| Weight | Default | Dimension |
|--------|---------|-----------|
| `AFF_WEIGHT_AUDIO` | 0.35 | Audio quality (DNSMOS normalized) |
| `AFF_WEIGHT_VIDEO` | 0.35 | Video quality (TOPIQ-NR) |
| `AFF_WEIGHT_EDITS` | 0.15 | Edit intentionality (scene cuts) |
| `AFF_WEIGHT_ENGAGEMENT` | 0.15 | Engagement rate bonus |

## Deployment (Railway)

### Docker

```bash
docker build -t affiliate-pipeline .
docker run --env-file .env -v $(pwd)/data:/data affiliate-pipeline
```

### Railway Setup

1. Connect the GitHub repo to Railway
2. Railway auto-detects the Dockerfile via `railway.toml`
3. Set environment variables in the Railway dashboard (all `AFF_*` vars)
4. Attach a 5GB volume mounted at `/data`
5. Upload your input CSV to `/data/input.csv`
6. Trigger a run (the service exits after processing)

**Recommended Railway resources:** 4GB RAM, 4 vCPU.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

## Project Structure

```
src/
  main.py                 # CLI entry point
  config.py               # All settings via environment variables
  models.py               # Data models (Pydantic)
  acquisition/            # Video downloading & metadata
  analyzers/              # Tier 1 (audio), Tier 2 (video), Tier 3 (Gemini)
  io/                     # CSV reading & writing
  pipeline/               # Orchestrator & batch processing
  scoring/                # Score aggregation & threshold logic
```
