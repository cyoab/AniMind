# AniMind Recommender

Phase-based CLI for generating LLM finetuning datasets over AniMind semantic IDs.

## Run with uv

```bash
cd /workspace/AniMind/recommender
uv sync
uv run animind-recommender run --phase prep --config ./config/recommender.toml
uv run animind-recommender run --phase train --config ./config/recommender.toml
```

## Phase: `prep`

`prep` builds a mixed-task prompt/completion training dataset:

- Task A: Semantic ID <-> metadata linking
- Task B: masked watch-sequence prediction
- Task C: template-based reasoning prompts from metadata overlap

Run-time visibility:

- Rich stage progress bar (`1/8` through `8/8`)
- Rich logs for loaded row counts and generated pool sizes
- Periodic scan logs during large `user_anime` passes

Primary outputs (under `[prep].out_dir`):

- `llm_prep_train.jsonl`
- `llm_prep_summary.json`
- `llm_prep_manifest.json` (if `write_manifest=true`)
- `llm_prep_train.parquet` (if `export_parquet=true`)

The generator is deterministic for a fixed config and `seed`.

## Input dependencies

`prep` expects:

- `source_db`: AniMind dataset SQLite (contains `anime` and `user_anime`)
- `semantic_ids_path`: tokenizer `semantic_ids.jsonl` artifact

Default paths in `config/recommender.toml` assume this repo layout:

- `/workspace/AniMind/output/anilist.sqlite`
- `/workspace/AniMind/recommender/data/semantic_ids/conservative/semantic_ids.jsonl`

## Config overview (`[prep]`)

- `target_examples`: total rows to generate (default `400000`)
- `split_task_a`, `split_task_b`, `split_task_c`: per-task ratio split (must sum to `1.0`)
- `task_b_*`: watch-sequence filtering and masking controls
- `rebuild`: if `false`, keep existing output JSONL when present
- `write_manifest`: write run/config manifest JSON

## Output row schema

Each JSONL row includes:

- `example_id`
- `task` (`A`, `B`, or `C`)
- `input`
- `output`
- `anime_ids`
- `user_id` (nullable)
- `meta` (template/mask metadata)

## Phase: `train`

`train` fine-tunes `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` in two ordered stages:

- Phase 1 (linking): Task A-weighted training
- Phase 2 (reasoning): Task B/C-weighted training

Key behaviors:

- Reuses tokenizer-style runtime checks for CUDA device + dtype resolution
- Extends tokenizer with semantic tokens and resizes model embeddings
- Optional warm-start for SID tokens from RQ-VAE codebook projection
- Mixes general instruction data (SlimOrca + OpenHermes) at configurable ratio
- English retention monitor via WikiText-2 perplexity every N steps
- Checkpoints saved every `train.checkpoint_every_steps`
- W&B integration (`train.wandb`)
- Dry-run mode for full wiring validation (`train.dry_run`)

Primary outputs (under `[train].out_dir`, or dry-run subdir):

- `phase1_linking/` checkpoints + best model
- `phase2_reasoning/` checkpoints + best model
- `final_model/`
- `tokenizer/`
- `train_run_summary.json`
- `train_config_snapshot.json`

### Dry-run first

Default config ships with `train.dry_run = true` so you can validate the full training pipeline before a full run.

```bash
cd /workspace/AniMind/recommender
uv sync
uv run animind-recommender run --phase train --config ./config/recommender.toml
```

On this host, CUDA and external network may be unavailable. In that case dry-run automatically:

- downgrades precision to `float32` when CUDA is unavailable
- uses `train.dry_run_model_name` fallback if base model retrieval fails
- falls back to synthetic general-mix rows and local English eval texts when dataset downloads fail

For full training, set:

- `train.dry_run = false`
- valid `HF_TOKEN` if needed for model access
- optional W&B settings under `[train.wandb]`
