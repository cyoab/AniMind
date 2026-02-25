# AniMind Dataset Builder

Modernized scraper pipeline for AniList/MAL-style interactions.

## Run with uv

```bash
cd /Users/yoab/Desktop/Projects/AniMind/dataset_builder
uv sync
uv run animind-dataset --out-dir ./output
```

## What it does

- Uses the legacy pacing rules from `original_scrapper.ipynb`:
  - 3s between club discovery pages
  - 4.2s between API requests
  - 120s cooldown retry on transient request failures
- Automatically resumes phase 2/3/4 from SQLite checkpoints
- Writes canonical data to SQLite (`anilist.sqlite`)
- Exports parquet snapshots for all sqlite tables (including resume/progress tables)
- Uses Rich logging/progress for terminal visibility
- Supports Kaggle bootstrap imports with idempotent inserts (`INSERT OR IGNORE`)
- Builds/updates a Jikan-backed anime catalog (`--phase anime`) and only inserts missing rows

## Generated schema

The pipeline writes one SQLite database at `output/anilist.sqlite` and parquet snapshots for each table.

### Core data tables

`clubs`
- `club_id INTEGER PRIMARY KEY`
- `members INTEGER NOT NULL`
- `discovered_at TEXT NOT NULL`

`users`
- `user_id INTEGER PRIMARY KEY`
- `username TEXT NOT NULL UNIQUE`
- `source_club_id INTEGER`
- `discovered_at TEXT NOT NULL`

`user_anime`
- `user_id INTEGER NOT NULL` (part of composite PK)
- `anime_id INTEGER NOT NULL` (part of composite PK)
- `score INTEGER NOT NULL`
- `watching_status INTEGER NOT NULL`
- `watched_episodes INTEGER NOT NULL`
- `scraped_at TEXT NOT NULL`

`anime`
- `anime_id INTEGER PRIMARY KEY`
- `name TEXT`
- `english_name TEXT`
- `score REAL`
- `genres TEXT`
- `synopsis TEXT`
- `type TEXT`
- `episodes INTEGER`
- `premiered TEXT`
- `studios TEXT`
- `source TEXT`
- `rating TEXT`
- `members INTEGER`
- `favorites INTEGER`
- `scored_by INTEGER`
- `imported_at TEXT NOT NULL`

### Progress/state tables

`club_scan_progress`
- `club_id INTEGER PRIMARY KEY`
- `scanned_at TEXT NOT NULL`

`club_member_progress`
- `club_id INTEGER PRIMARY KEY`
- `last_page INTEGER NOT NULL DEFAULT 0`
- `completed INTEGER NOT NULL DEFAULT 0`
- `updated_at TEXT NOT NULL`

`user_animelist_progress`
- `user_id INTEGER PRIMARY KEY`
- `last_page INTEGER NOT NULL DEFAULT 0`
- `completed INTEGER NOT NULL DEFAULT 0`
- `updated_at TEXT NOT NULL`

`anime_catalog_progress`
- `anime_id INTEGER PRIMARY KEY`
- `last_status INTEGER NOT NULL DEFAULT 0`
- `completed INTEGER NOT NULL DEFAULT 0`
- `updated_at TEXT NOT NULL`

`state`
- `key TEXT PRIMARY KEY`
- `value TEXT NOT NULL`

## Dataset snapshot (2026-02-25)

Current `output/anilist.sqlite` size: **~8.0 GiB**.

SQLite row counts:

| table | rows |
|---|---:|
| clubs | 9,138 |
| club_scan_progress | 1,978 |
| club_member_progress | 1,978 |
| users | 373,367 |
| user_animelist_progress | 325,769 |
| user_anime | 109,224,672 |
| anime | 17,562 |
| anime_catalog_progress | 0 |
| state | 3 |

Parquet snapshot sizes (`output/*.parquet`):

| file | rows | size |
|---|---:|---:|
| `user_anime.parquet` | 109,224,672 | 327,618,963 B (~312.4 MiB) |
| `users.parquet` | 373,367 | 5,264,686 B (~5.0 MiB) |
| `user_animelist_progress.parquet` | 325,769 | 1,595,072 B (~1.5 MiB) |
| `anime.parquet` | 17,562 | 635,402 B (~620.5 KiB) |
| `clubs.parquet` | 9,138 | 76,774 B (~75.0 KiB) |
| `club_member_progress.parquet` | 1,978 | 27,845 B (~27.2 KiB) |
| `state.parquet` | 3 | 1,705 B (~1.7 KiB) |

Notes:
- Counts come from the live SQLite DB.
- Parquet sizes are from the most recent export files and can lag if scraping is still running.

## Resume and status

Show saved progress:

```bash
uv run animind-dataset --out-dir ./output --status-only
```

Bootstrap with Kaggle (safe to rerun):

```bash
uv run animind-dataset --out-dir ./output --bootstrap-kaggle
```

Note: `kagglehub` needs Kaggle credentials configured in your environment.

Use a local dataset folder instead of downloading:

```bash
uv run animind-dataset --out-dir ./output --bootstrap-kaggle --bootstrap-path /path/to/dataset
```

Bootstrap then continue scraping:

```bash
uv run animind-dataset --out-dir ./output --bootstrap-kaggle --phase users
uv run animind-dataset --out-dir ./output --phase animelist
uv run animind-dataset --out-dir ./output --phase anime
```

Club parsing concurrency test (phase 2 only):

```bash
uv run animind-dataset --out-dir ./output --phase users --club-workers 4 --club-limit 200
```

Use local Jikan Docker for higher throughput:

```bash
docker compose -f ./docker/jikan-local.compose.yml up -d
uv run animind-dataset --out-dir ./output --phase users --jikan-base-url http://localhost:8080/v4 --api-delay-seconds 0.05 --club-workers 8
```

If you need to benchmark upper limits locally, edit `docker/jikan-local.compose.yml` and tune `RATE_LIMIT` / Redis cache settings.

Resume a specific phase:

```bash
uv run animind-dataset --out-dir ./output --phase users
uv run animind-dataset --out-dir ./output --phase animelist
uv run animind-dataset --out-dir ./output --phase anime
```

Reset only checkpoints (keeps scraped data):

```bash
uv run animind-dataset --out-dir ./output --reset-resume
```

## Export existing DB only

```bash
uv run animind-dataset --out-dir ./output --export-only
```
