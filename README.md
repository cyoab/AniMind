# AniMind

AniMind has three pipeline modules:

- `dataset_builder`: scrapes/builds AniList-style user + anime data into SQLite/parquet
- `tokenizer`: builds anime text/embeddings/RQ-VAE/Semantic IDs
- `recommender`: builds LLM finetuning datasets from Semantic IDs and watch history

## Dataset Builder

```bash
cd /workspace/AniMind/dataset_builder
uv sync
uv run animind-dataset scrape --out-dir ../output --phase all
```

See `dataset_builder/README.md` for resume, bootstrap, and phase controls.

## Tokenizer

```bash
cd /workspace/AniMind/tokenizer
uv sync
uv run animind-tokenizer run --phase prep --config ./config/tokenizer.toml
uv run animind-tokenizer run --phase embed --config ./config/tokenizer.toml
uv run animind-tokenizer run --phase rqvae --config ./config/tokenizer.toml
uv run animind-tokenizer run --phase tokenize --config ./config/tokenizer.toml
```

See `tokenizer/README.md` for phase details and artifacts.

## Recommender Prep Dataset

```bash
cd /workspace/AniMind/recommender
uv sync
uv run animind-recommender run --phase prep --config ./config/recommender.toml
```

Output files are written to the configured `out_dir`:

- `llm_prep_train.jsonl`
- `llm_prep_summary.json`
- `llm_prep_manifest.json`
- `llm_prep_train.parquet` (optional)
