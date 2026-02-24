# AniMind - Anime Recommendation System

## Dataset Build (New Pipeline)

```bash
cd /Users/yoab/Desktop/Projects/AniMind/dataset_builder
uv sync
docker compose -f ./docker/jikan-local.compose.yml up -d
uv run animind-dataset build --phase all --target-users 10000 --out-dir ./output --include-nsfw
```

Two-pass option:

```bash
uv run animind-dataset build --phase interactions --target-users 10000 --out-dir ./output --include-nsfw
uv run animind-dataset build --phase anime --run-id <run_id> --target-users 10000 --out-dir ./output --include-nsfw
```

See `/Users/yoab/Desktop/Projects/AniMind/dataset_builder/README.md` for all runtime profile and discovery-source options.
