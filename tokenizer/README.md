# AniMind Tokenizer

Tokenizer pipeline that prepares AniMind datasets for model-training token generation.

## Run with uv

```bash
cd /Users/yoab/Desktop/Projects/AniMind/tokenizer
uv sync
uv run animind-tokenizer run --phase prep --config ./config/tokenizer.toml
```

## Config-first workflow

The tokenizer now uses a TOML config file to avoid long CLI flag lists.

Default config path:

- `./config/tokenizer.toml`

Required sections:

- `[prep]` for prep-phase settings
- `[embedd]` for embedding-phase settings (intentionally named `embedd` for extensibility compatibility)

Example:

```bash
[prep]
source_db = "../../output/anilist.sqlite"
out_dir = "../output"
rebuild = true
export_parquet = true
limit = 0

[embedd]
tokenizer_db = "../output/tokenizer.sqlite"
rebuild = true
limit = 0
model_name = "tencent/KaLM-Embedding-Gemma3-12B-2511"
batch_size = 8
max_length = 2048
device = "cuda"
normalize = true
```

## Phase: `prep`

Run:

```bash
uv run animind-tokenizer run --phase prep --config ./config/tokenizer.toml
```

`prep` outputs:

- SQLite table: `anime_prep`
- Parquet file: `anime_prep.parquet` (if `export_parquet=true`)

## Phase: `embed`

`embed` reads `anime_text` from `anime_prep`, computes embeddings using the configured Hugging Face model, and stores vectors in SQLite.

- Input table: `anime_prep`
- Output table: `anime_embeddings`
- Default model: `tencent/KaLM-Embedding-Gemma3-12B-2511`

Run on a GPU pod:

```bash
uv run animind-tokenizer run \
  --phase embed \
  --config ./config/tokenizer.toml
```

`anime_embeddings` schema:

- `anime_id INTEGER PRIMARY KEY`
- `embedding BLOB NOT NULL` (float32 vector bytes)
- `embedding_dim INTEGER NOT NULL`
- `model_name TEXT NOT NULL`
- `embedded_at TEXT NOT NULL`
- `source_prepared_at TEXT`

## Status and outputs

After a successful `prep` + `embed` run:

- `output/tokenizer.sqlite` exists.
- `anime_prep` contains cleaned anime metadata and `anime_text`.
- `anime_embeddings` contains one vector per `anime_id`.
- `anime_text` is single-line, deterministic key-value content per anime record.
