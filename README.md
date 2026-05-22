# Creator Screener

A multi-stage pipeline for vetting TikTok and Instagram creators for affiliate partnerships. Takes a list of handles (or a campaign brief), pulls engagement data, screens for content fit, scores TikTok video production quality, and generates personalized outreach copy.

## How It Works

```
Input: CSV of handles  OR  campaign brief
  |
  v
[1] Discovery (optional)
    - LLM suggests creators matching a niche
    - yt-dlp verifies each TikTok handle exists + pulls baseline metrics
  |
  v
[2] Enrichment
    - TikTok: yt-dlp pulls 10 recent videos -> views, likes, comments, shares, saves
    - Instagram: RapidAPI pulls profile + 12 recent posts -> engagement metrics
    - Computes engagement rate per platform
  |
  v
[3] Creator Scoring (cheap pre-filter)
    - Composite of engagement, follower quality, consistency, authenticity
    - Drops creators below AFF_CREATOR_SCORE_MIN before any expensive work
  |
  v
[4] Content-Fit Review (LLM)
    - Gemini reads recent captions / video descriptions
    - Decides if the creator's niche matches the campaign
  |
  v
[5] Video Quality Pipeline (TikTok only)
    - Tier 1 (free): Silero VAD + DNSMOS audio scoring
    - Tier 2 (free): TOPIQ-NR aesthetic/technical + PySceneDetect edits
    - Tier 3 (cheap): Gemini Flash on key frames for borderline cases only
  |
  v
[6] Outreach Generation
    - Pulls fresh IG captions + TikTok descriptions
    - Gemini writes a personalized opener + closer per creator
    - Wraps them in a templated email + DM
  |
  v
Output: scored CSV with rankings + ready-to-send email/DM bodies
```

## Required Services

| Service | Used for | Cost |
|---------|----------|------|
| [OpenRouter](https://openrouter.ai) | Gemini Flash (content-fit, frame scoring, outreach personalization) | ~$0.01 per creator processed |
| [RapidAPI: Instagram Scraper Stable](https://rapidapi.com/thetechguy32744/api/instagram-scraper-stable-api) | Instagram profile + post enrichment | Tiered, basic plan ~$10/mo |

### Optional

| Service | When you need it |
|---------|------------------|
| Slack (bot + app token) | Trigger pipeline by uploading a CSV to a Slack channel (`--mode slack`) |
| Firecrawl | Legacy Instagram scraper fallback |
| Supabase | Persistent storage of scored creators across runs |

## Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg in PATH

### Setup

```bash
git clone https://github.com/mitchell-create/creator-screener.git
cd creator-screener

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add at minimum:
#   AFF_OPENROUTER_API_KEY
#   AFF_RAPIDAPI_KEY
```

### Input CSV Format

Create `data/input.csv` with at minimum a `profile_url` column. Optional columns: `engagement_rate`, `followers`.

```csv
profile_url,engagement_rate,followers
https://www.tiktok.com/@username1,5.2,50000
@username2,3.8,120000
```

Accepted column aliases: `profile_url` / `url` / `tiktok_url` / `link`, `engagement_rate` / `er`, `followers` / `follower_count`.

See [data/sample_input.csv](data/sample_input.csv) for a working example.

### Run

```bash
# Video quality pipeline (TikTok)
python -m src.main --input data/input.csv --output data/output.csv

# Limit to first N
python -m src.main -i data/input.csv -o data/output.csv --limit 10

# Modash-style pipeline (IG-first input -> TikTok filter -> creator + content-fit scoring)
python -m src.pipeline.modash_pipeline --input data/modash_export.csv --output data/scored.csv

# Generate personalized outreach for scored creators
python -m src.outreach.generate_messages \
    --input data/scored.csv \
    --output data/outreach.csv \
    --refresh   # re-fetch live IG + TikTok content for richer personalization
```

### Output

Two CSV files from the video quality pipeline:
- `output.csv` — all creators with scores, pass/fail, rejection reasons
- `output_passed.csv` — only creators that passed all tiers

Output columns include: `profile_url`, `engagement_rate`, `followers`, `videos_analyzed`, `avg_audio_score`, `avg_video_aesthetic`, `avg_video_technical`, `avg_scene_cuts`, `gemini_avg_score`, `overall_score`, `passed`, `rejection_reason`, `rejection_tier`.

The outreach generator outputs a CSV with `email_subject`, `email_body`, and `dm_body` per creator.

## Configuration

All settings are configured via `AFF_*` environment variables. See [.env.example](.env.example) for the full list.

### Key Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AFF_VAD_SPEECH_MIN_PCT` | 10.0 | Minimum speech percentage to pass Tier 1 |
| `AFF_DNSMOS_OVRL_MIN` | 2.5 | Minimum audio quality (DNSMOS, 1–5 scale) |
| `AFF_DOVER_OVERALL_MIN` | 0.45 | Minimum video quality score (0–1 scale) |
| `AFF_DOVER_AESTHETIC_MIN` | 0.4 | Minimum aesthetic score (0–1 scale) |
| `AFF_DOVER_TECHNICAL_MIN` | 0.4 | Minimum technical score (0–1 scale) |
| `AFF_MIN_SCENE_CUTS` | 2 | Minimum scene cuts per video |
| `AFF_TIER_FAILURE_RATE` | 0.5 | Fraction of videos that must fail to reject a creator |
| `AFF_GEMINI_MIN_SCORE` | 6.0 | Minimum Gemini Tier 3 score (1–10 scale) |
| `AFF_CREATOR_SCORE_MIN` | 0.0 | Drop creators below this composite score (0 = disabled) |

### Scoring Weights

Video quality composite (per-video):

| Weight | Default | Dimension |
|--------|---------|-----------|
| `AFF_WEIGHT_AUDIO` | 0.35 | Audio quality (DNSMOS, normalized) |
| `AFF_WEIGHT_VIDEO` | 0.35 | Video quality (TOPIQ-NR) |
| `AFF_WEIGHT_EDITS` | 0.15 | Edit intentionality (scene cuts) |
| `AFF_WEIGHT_ENGAGEMENT` | 0.15 | Engagement rate bonus |

Creator score composite (across-platform):

| Weight | Default | Dimension |
|--------|---------|-----------|
| `AFF_CREATOR_WEIGHT_ENGAGEMENT` | 0.35 | Engagement rate vs. follower count |
| `AFF_CREATOR_WEIGHT_FOLLOWER_QUALITY` | 0.25 | Proxies for real vs. inflated followers |
| `AFF_CREATOR_WEIGHT_CONSISTENCY` | 0.20 | Variance across recent posts |
| `AFF_CREATOR_WEIGHT_AUTHENTICITY` | 0.20 | Verification, bio, post-mix signals |

## Deployment (Railway)

### Docker

```bash
docker build -t creator-screener .
docker run --env-file .env -v $(pwd)/data:/data creator-screener
```

### Railway Setup

1. Connect the GitHub repo to Railway
2. Railway auto-detects the Dockerfile via `railway.toml`
3. Set environment variables in the Railway dashboard (all `AFF_*` vars)
4. Attach a 5GB volume mounted at `/data`
5. Upload your input CSV to `/data/input.csv`
6. Trigger a run (the service exits after processing)

**Recommended resources:** 4GB RAM, 4 vCPU.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## Project Structure

```
src/
  main.py                    # CLI entry point — video quality pipeline
  config.py                  # All settings via AFF_* env vars
  models.py / models_creator.py  # Pydantic data models
  acquisition/               # yt-dlp + RapidAPI scrapers, video download
  analyzers/                 # Tier 1 (audio), Tier 2 (video), Tier 3 (Gemini), content-fit, brand-fit
  discovery/                 # LLM-driven creator suggestion + handle verification
  pipeline/
    orchestrator.py          # Video quality pipeline
    modash_pipeline.py       # IG-first pipeline (Modash exports -> scored CSV)
    enrichment.py            # Multi-platform profile scraping + creator scoring
    batch_processor.py       # Concurrency, batching, progress tracking
  scoring/                   # Creator signals, affiliate readiness, thresholds
  outreach/                  # Personalized email + DM generation (LLM-driven)
  io/                        # CSV reader/writer, Supabase persistence
  slack_bot.py               # Slack-triggered pipeline runs
```

## Notes

- Brand-specific copy for outreach generation lives in [src/outreach/generate_messages.py](src/outreach/generate_messages.py) — edit the `EMAIL_BRAND_PITCH` / `DM_BRAND_PITCH` / `EMAIL_CTA` constants for your campaign.
- TikTok hashtag/discovery-page scraping is blocked, so discovery mode relies on LLM-suggested handles verified against live profiles.
- The video quality pipeline only works on TikTok; Instagram is metadata-only (no video download).
