-- Migration: Add creator enrichment columns to affiliates table
-- Run this against your Supabase project to support the new creator scoring pipeline.
--
-- These columns store data from Tier 0 (creator enrichment) — scraped profile
-- metrics and computed creator-level scores that run before video analysis.

-- Creator scoring signals
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_score float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_engagement_score float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_follower_quality float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_consistency float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_authenticity float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_brand_fit float8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_tier text;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS creator_flags text;

-- Scraped TikTok data
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS tiktok_followers_scraped int8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS tiktok_following int8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS tiktok_total_likes int8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS tiktok_engagement_rate_scraped float8;

-- Scraped Instagram data
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS instagram_handle text;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS instagram_followers int8;
ALTER TABLE affiliates ADD COLUMN IF NOT EXISTS instagram_engagement_rate float8;

-- Index for filtering by creator score and tier
CREATE INDEX IF NOT EXISTS idx_affiliates_creator_score ON affiliates (creator_score);
CREATE INDEX IF NOT EXISTS idx_affiliates_creator_tier ON affiliates (creator_tier);

-- Enrichment stats on runs table
ALTER TABLE affiliate_runs ADD COLUMN IF NOT EXISTS enriched int4 DEFAULT 0;
ALTER TABLE affiliate_runs ADD COLUMN IF NOT EXISTS enrichment_failed int4 DEFAULT 0;
ALTER TABLE affiliate_runs ADD COLUMN IF NOT EXISTS pre_filtered int4 DEFAULT 0;
