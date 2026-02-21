# AniMind Dataset Builder

A terminal-first Jikan v4 scraper that creates a local anime dataset with:

- user anime lists (`watching`, `completed`, `on_hold`, `dropped`, `plan_to_watch`)
- completed ratings (`status=completed` and `score>0`)
- anime enrichment (full details, statistics, staff, and reviews)

## Quick Start

```bash
cd /Users/yoab/Desktop/Projects/AniMind/dataset_builder
uv sync
uv run animind-dataset build --target-users 10000 --out-dir ./output --include-nsfw
uv run animind-dataset stats --run-id <run_id> --out-dir ./output
```

## Outputs

- `users.parquet`
- `user_anime_list.parquet`
- `completed_ratings.parquet`
- `anime.parquet`
- `anime_statistics.parquet`
- `anime_staff.parquet`
- `anime_reviews.parquet`
- `run_manifest.json`

SQLite state is stored in `<out-dir>/state.sqlite3`.
