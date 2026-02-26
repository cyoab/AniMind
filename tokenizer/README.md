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
- `[rqvae]` for RQ-VAE training settings
- `[tokenize]` for Semantic ID/token vocabulary generation

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
env_file = "../.env"
rebuild = true
limit = 0
model_name = "tencent/KaLM-Embedding-Gemma3-12B-2511"
batch_size = 8
max_length = 2048
device = "cuda"
precision = "auto"
hf_token = ""
hf_token_env = "HF_TOKEN"
normalize = true

[rqvae]
tokenizer_db = "../output/tokenizer.sqlite"
out_dir = "../output/rqvae"
rebuild = true
limit = 0
device = "auto"
seed = 42
batch_size = 256
epochs = 40
num_workers = 4
val_ratio = 0.05
lr = 0.00004
adam_beta1 = 0.5
adam_beta2 = 0.9
warmup_steps = 1000
latent_dim = 256
rq_levels = 8
codebook_size = 2048
commitment_beta = 0.25
ema_decay = 0.99
ema_eps = 0.00001
restart_unused_codes = true
amp = true
checkpoint_every = 1
encoder_hidden_dim = 1024
decoder_hidden_dim = 1024
dry_run = false
dry_run_limit = 512
dry_run_epochs = 1
dry_run_batch_size = 32
dry_run_num_workers = 0
dry_run_out_subdir = "dry_run"
env_file = "../.env"
wandb_enabled = false
wandb_mode = "offline"
wandb_project = ""
wandb_entity = ""
wandb_api_key = ""
wandb_run_name = ""
wandb_group = ""
wandb_tags = []
wandb_project_env = "WANDB_PROJECT"
wandb_entity_env = "WANDB_ENTITY"
wandb_api_key_env = "WANDB_API_KEY"
resume_from = ""
resume_strict = true

[tokenize]
tokenizer_db = "../output/tokenizer.sqlite"
source_db = "../../output/anilist.sqlite"
rqvae_checkpoint = "../output/rqvae_conservative/rqvae_best.pt"
out_dir = "../output/tokenize"
rebuild = true
limit = 0
device = "auto"
batch_size = 512
write_db_tables = false
special_tokens = ["<anime_start>", "<anime_end>", "<watch>", "<liked>", "<disliked>", "<dropped>"]
semantic_id_concat = true
semantic_id_separator = ""
cluster_sample_size = 10
cluster_min_bucket = 20
cluster_random_seed = 42
recall_k = 20
recall_min_support = 10
recall_positive_score_min = 7
recall_completed_status = 2
recall_max_queries = 500
recall_max_rows = 0
recall_seed = 42
dry_run = false
dry_run_limit = 512
dry_run_out_subdir = "dry_run"
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

Precision guidance:

- `embedd.precision = "auto"` uses `bfloat16` on CUDA when available, otherwise `float32`.
- Avoid `float16` unless explicitly required; it is more likely to produce non-finite embeddings on large models.
- The loader uses current Transformers API (`dtype=` and `token=`), with compatibility fallback only if needed.

Hugging Face token support:

- Set `embedd.env_file` and define `HF_TOKEN=...` (or `HUGGINGFACE_HUB_TOKEN=...`) in that file.
- Or set `embedd.hf_token` directly (not recommended for committed configs).
- `embedd.hf_token_env` lets you choose a custom env variable name.

## Phase: `rqvae`

`rqvae` trains a shared-codebook Residual-Quantized VAE over `anime_embeddings` and writes checkpoints and metrics.

- Input table: `anime_embeddings`
- Outputs directory: `output/rqvae` (configurable)
- EMA codebook updates with random restarts for unused codes
- Straight-through estimator: `z_quantized = z + (z_discrete - z).detach()`
- Rich pipeline progress for load, split, init, train/val, checkpoint, and finalize stages
- Optional W&B tracking (`offline`/`online`) configured in `[rqvae]`

Run:

```bash
uv run animind-tokenizer run \
  --phase rqvae \
  --config ./config/tokenizer.toml
```

Artifacts:

- `rqvae_best.pt`
- `rqvae_last.pt`
- `rqvae_config.json`
- `rqvae_metrics.jsonl`
- `code_usage.csv`

Resume support:

- Set `resume_from = "last"` (or `"best"` / absolute checkpoint path) to resume training after interruption.
- Set `resume_strict = true` to fail fast if the checkpoint is missing.
- `rqvae_last.pt` is always saved at training end, even if `checkpoint_every` does not divide the final epoch.

Evaluation loading:

- Use `load_rqvae_for_eval(checkpoint_path=Path(...), device="cpu|cuda")` from `animind_tokenizer.rqvae` to reconstruct the model and load weights for inference/eval.

### W&B + `.env`

If `wandb_enabled=true`, the trainer resolves `project`, `entity`, and optional API key from:

1. direct `[rqvae]` values (`wandb_project`, `wandb_entity`, `wandb_api_key`)
2. env keys configured in `[rqvae]` (`wandb_*_env`) from `env_file`
3. process environment variables

For offline tracking:

```bash
WANDB_PROJECT=animind-tokenizer
WANDB_ENTITY=my-team
```

in your `.env` and set `wandb_mode = "offline"`.

### Dry-run

Set `dry_run = true` in `[rqvae]` to execute a short safety run that:

- caps rows and epochs
- writes to `out_dir/<dry_run_out_subdir>`
- validates checkpoint/metrics artifact integrity at the end

## Status and outputs

After a successful `prep` + `embed` run:

- `output/tokenizer.sqlite` exists.
- `anime_prep` contains cleaned anime metadata and `anime_text`.
- `anime_embeddings` contains one vector per `anime_id`.
- `anime_text` is single-line, deterministic key-value content per anime record.

## Phase: `tokenize`

`tokenize` loads a trained RQ-VAE checkpoint and converts each anime embedding into a hierarchical Semantic ID.

- Input tables: `anime_prep`, `anime_embeddings`
- Input checkpoint: `rqvae_best.pt` (or another compatible checkpoint)
- Output directory: `output/tokenize` (configurable)
- Builds vocabulary tokens like `<L1_47>`, `<L2_203>`, ...
- Adds special tokens like `<anime_start>`, `<watch>`, `<liked>`, ...
- Writes lookup artifacts (`Semantic ID ↔ anime title ↔ anime_id`)
- Runs validation helpers:
  - shared-first-token cluster inspection
  - recall@K evaluation against watch-history proxy similarity from `user_anime`
  - `recall_max_rows` can cap `user_anime` scan size; dry-run mode automatically caps this for fast checks

Run:

```bash
uv run animind-tokenizer run \
  --phase tokenize \
  --config ./config/tokenizer.toml
```

Artifacts:

- `semantic_ids.jsonl`
- `semantic_vocab.json`
- `special_tokens.json`
- `semantic_lookup.tsv`
- `semantic_id_to_anime.jsonl`
- `cluster_inspection.jsonl`
- `cluster_summary.json`
- `recall_report.json`
- `tokenize_run_summary.json`
