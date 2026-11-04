# Extraction design notes

**Topic:** safe restraint and immobilization techniques for infants and young children (clinical holding, needle procedures, imaging, spinal and fracture immobilization, dental protective stabilization, safety risks, alternatives, neonatal positioning, examination and wound care). Vehicle child restraints (car seats) are out of scope.

**Queries:** 25 English queries in `config/queries.csv` (`query_id, query, subtopic, query_language`). The `subtopic` describes the query, not the video; a video's true subtopic is decided later.

**Why these tools**
* YouTube search + metadata: `yt-dlp` - free, no API key, rich fields (views, likes, tags, upload date).
* YouTube transcripts: `youtube-transcript-api` - reads existing captions, manual preferred over auto-generated.
* TikTok: ScrapeCreators keyword search - reliable and credit-based. Its response already contains stats, hashtags, region and a caption (WebVTT) URL, so TikTok transcripts cost no extra credits.

**Caching (`data/cache/`)** - namespaces: `yt_search`, `yt_video`, `yt_transcript`, `tt_search`, `tt_caption`.
A video returned by several queries is fetched once. Permanent outcomes (no captions, private video) are cached; temporary ones (IP block, network error) are not, so they are retried on the next run.

**Reliability** - retries with exponential backoff; per-video failures never stop a query; a failed video still gets a row with `metadata_status` explaining why; CSVs are written atomically; incomplete queries are saved as `.partial.csv`.

**Credits** - 25 queries x 2 TikTok pages = 50 credits. `max_credits_per_run` (60) is a hard stop and `credit_reserve` (5) is never spent.

**Preprocessing**
1. *Merge* - all `data/raw/Q*.csv` (a complete file is preferred over a partial one for the same query) -> `master_raw.csv`.
2. *Deduplicate* - one row per `(platform, video_id)`. The kept row is the most complete (metadata ok > transcript ok > best rank). Which queries found the video is preserved in `query_ids`, `query_count`, `best_query_rank`, `query_subtopics`; the long table is `query_video_map.csv`.
3. *Clean* - fixes text encoding and control characters, converts counts and dates, adds `video_uid`, `language_clean`, `country_clean`, `has_transcript`. Rows are dropped only for four documented reasons (`metadata_not_ok`, `invalid_url`, `invalid_duration`, `no_text_at_all`) and each drop is listed in `dropped_records.csv`.

**Not included yet (planned fallbacks)** - speech-to-text for videos without captions; YouTube Data API for channel country; ScrapeCreators profile endpoint for TikTok creator country.
