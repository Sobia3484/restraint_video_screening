# Data dictionary

## 1. Per-query CSVs (`data/raw/Qxxx_<slug>.csv`)

| Column | Type | Description |
|---|---|---|
| query_id | str | e.g. `Q009` - links to `config/queries.csv` |
| query_text | str | The search text used |
| platform | str | `youtube` or `tiktok` |
| query_rank | int | Position in that platform's results for this query (1 = top) |
| video_id | str | Platform video id (dedupe key together with `platform`) |
| url | str | Canonical video URL |
| title | str | YouTube title. **TikTok: caption text is stored here** |
| description | str | YouTube description. TikTok: empty (single caption field) |
| creator_name | str | Channel / TikTok nickname |
| creator_id | str | YouTube `channel_id` / TikTok handle |
| creator_url | str | Channel or profile URL |
| creator_followers | int | Subscribers / followers when available |
| duration_seconds | float | Video length in seconds |
| upload_date | date | `YYYY-MM-DD` (UTC) |
| views, likes, comments | int | Engagement counts at retrieval time (likes may be hidden on YouTube, then blank) |
| shares, saves | int | TikTok only |
| hashtags | str | Pipe-separated, lower-case, from caption/title/description |
| tags | str | YouTube tags, pipe-separated |
| category | str | YouTube category |
| language | str | Platform-reported language; often blank or `un` |
| country_code | str | Only when the platform reports it (TikTok video `region`); never inferred |
| country_source | str | Where `country_code` came from |
| transcript | str | Full transcript text |
| transcript_status | str | `ok`, `no_transcript`, `disabled`, `video_unavailable`, `age_restricted`, `blocked`, `skipped_blocked`, `no_caption_available`, `empty_caption`, `caption_download_failed`, `not_requested`, `error:<Type>` |
| transcript_source | str | `youtube_captions`, `tiktok_auto_caption`, `tiktok_caption` |
| transcript_language | str | Language code of the transcript |
| transcript_is_auto | bool | Auto-generated captions? |
| transcript_word_count | int | Words in transcript |
| metadata_status | str | `ok`, `unavailable: ...`, `failed: ...` |
| thumbnail_url | str | Thumbnail/cover URL (TikTok URLs expire) |
| retrieved_at_utc | str | Extraction timestamp |

`data/raw/_manifest.csv` logs each query run: rows per platform, transcripts found, TikTok credits spent, status (`complete`/`partial`) and notes.

## 2. `data/intermediate/`

| File | Rows | Description |
|---|---|---|
| `master_raw.csv` | one per (query, video) | All per-query CSVs stacked; adds `source_file` |
| `master_deduplicated.csv` | one per unique video | Duplicates collapsed; query columns replaced by the four below |
| `query_video_map.csv` | one per (query, video) | Which query found which video at which rank (`query_id, query_text, platform, video_id, query_rank, subtopic`) |
| `master_cleaned.csv` | one per kept video | Cleaned table used for the next stages |
| `dropped_records.csv` | one per dropped video | `video_uid, platform, video_id, url, metadata_status, reason` |
| `preprocessing_report.json` | | Counts for every step |

Columns added by deduplication:

| Column | Description |
|---|---|
| query_ids | Pipe-separated ids of every query that found the video |
| query_count | Number of distinct queries that found it |
| best_query_rank | Best (lowest) rank across those queries |
| query_subtopics | Pipe-separated subtopics of those queries |

Columns added or changed by cleaning:

| Column | Description |
|---|---|
| video_uid | `platform:video_id`, unique key |
| has_description | TikTok is blank (it has one caption field only) |
| language_clean | Primary language code or `unknown` |
| country_clean | Two-letter code or `Unknown` |
| has_transcript | True when `transcript_status = ok` and the text is not empty |
| transcript_language_clean | Cleaned transcript language |
| transcript_status = `corrupt_encoding` | Garbled TikTok caption that could not be repaired; transcript blanked |

Drop reasons: `metadata_not_ok`, `invalid_url`, `invalid_duration`, `no_text_at_all`.
