from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .prep import PrepConfig
from .train_types import (
    EvalConfig,
    GeneralMixConfig,
    LoRAConfig,
    PhaseConfig,
    TokenExtensionConfig,
    TokenWarmStartConfig,
    TrainConfig,
    WandbConfig,
)

_ALLOWED_MASK_MODES = {"last", "random"}
_ALLOWED_WANDB_MODES = {"offline", "online", "disabled"}
_ALLOWED_GENERAL_SOURCES = {"SlimOrca", "OpenHermes"}
_ALLOWED_PRECISIONS = {"auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}


def build_prep_config(config_path: Path) -> PrepConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="prep")

    source_db = _resolve_path(
        config_path=config_path,
        raw_path=section.get("source_db", "../../output/anilist.sqlite"),
    )
    semantic_ids_path = _resolve_path(
        config_path=config_path,
        raw_path=section.get(
            "semantic_ids_path", "../data/semantic_ids/conservative/semantic_ids.jsonl"
        ),
    )
    out_dir = _resolve_path(config_path=config_path, raw_path=section.get("out_dir", "../output"))

    rebuild = bool(section.get("rebuild", True))
    seed = int(section.get("seed", 42))
    target_examples = int(section.get("target_examples", 400_000))
    split_task_a = float(section.get("split_task_a", 0.50))
    split_task_b = float(section.get("split_task_b", 0.25))
    split_task_c = float(section.get("split_task_c", 0.25))
    max_task_a_templates_per_anime = int(section.get("max_task_a_templates_per_anime", 12))
    task_b_min_history = int(section.get("task_b_min_history", 4))
    task_b_max_history = int(section.get("task_b_max_history", 30))

    raw_mask_modes = section.get("task_b_mask_modes", ["last", "random"])
    if not isinstance(raw_mask_modes, list) or not raw_mask_modes:
        raise RuntimeError(
            "Invalid prep.task_b_mask_modes in config: must be a non-empty array of strings."
        )
    task_b_mask_modes = [str(mode).strip().lower() for mode in raw_mask_modes if str(mode).strip()]
    if not task_b_mask_modes:
        raise RuntimeError(
            "Invalid prep.task_b_mask_modes in config: must contain at least one non-empty value."
        )
    invalid_modes = sorted(set(task_b_mask_modes) - _ALLOWED_MASK_MODES)
    if invalid_modes:
        raise RuntimeError(
            "Invalid prep.task_b_mask_modes in config: "
            f"unsupported values {invalid_modes}; allowed={sorted(_ALLOWED_MASK_MODES)}"
        )

    task_b_positive_score_min = int(section.get("task_b_positive_score_min", 7))
    raw_statuses = section.get("task_b_allowed_statuses", [1, 2, 3, 4, 6])
    if not isinstance(raw_statuses, list) or not raw_statuses:
        raise RuntimeError(
            "Invalid prep.task_b_allowed_statuses in config: must be a non-empty integer array."
        )
    task_b_allowed_statuses = [int(status) for status in raw_statuses]

    export_parquet = bool(section.get("export_parquet", False))
    write_manifest = bool(section.get("write_manifest", True))

    if target_examples < 1:
        raise RuntimeError("Invalid prep.target_examples in config: must be >= 1")
    if max_task_a_templates_per_anime < 1:
        raise RuntimeError("Invalid prep.max_task_a_templates_per_anime in config: must be >= 1")
    if task_b_min_history < 2:
        raise RuntimeError("Invalid prep.task_b_min_history in config: must be >= 2")
    if task_b_max_history < task_b_min_history:
        raise RuntimeError(
            "Invalid prep.task_b_max_history in config: must be >= prep.task_b_min_history"
        )
    if task_b_positive_score_min < 0:
        raise RuntimeError("Invalid prep.task_b_positive_score_min in config: must be >= 0")
    if any(status < 1 for status in task_b_allowed_statuses):
        raise RuntimeError("Invalid prep.task_b_allowed_statuses in config: values must be >= 1")
    if not (0 < split_task_a < 1) or not (0 < split_task_b < 1) or not (0 < split_task_c < 1):
        raise RuntimeError("Invalid prep split values in config: each split must be > 0 and < 1")

    split_total = split_task_a + split_task_b + split_task_c
    if abs(split_total - 1.0) > 1e-8:
        raise RuntimeError(
            f"Invalid prep splits in config: split_task_a+split_task_b+split_task_c must equal 1.0, got {split_total:.6f}"
        )

    return PrepConfig(
        source_db=source_db,
        semantic_ids_path=semantic_ids_path,
        out_dir=out_dir,
        rebuild=rebuild,
        seed=seed,
        target_examples=target_examples,
        split_task_a=split_task_a,
        split_task_b=split_task_b,
        split_task_c=split_task_c,
        max_task_a_templates_per_anime=max_task_a_templates_per_anime,
        task_b_min_history=task_b_min_history,
        task_b_max_history=task_b_max_history,
        task_b_mask_modes=task_b_mask_modes,
        task_b_positive_score_min=task_b_positive_score_min,
        task_b_allowed_statuses=task_b_allowed_statuses,
        export_parquet=export_parquet,
        write_manifest=write_manifest,
    )


def _load_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    with config_path.open("rb") as file:
        parsed = tomllib.load(file)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid config format: {config_path}")
    return parsed


def _require_section(doc: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = doc.get(section_name)
    if not isinstance(section, dict):
        raise RuntimeError(f"Missing required config section: [{section_name}]")
    return section


def _resolve_path(config_path: Path, raw_path: Any) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def build_train_config(config_path: Path) -> TrainConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="train")

    base_model = str(section.get("base_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")).strip()
    train_jsonl = _resolve_path(
        config_path=config_path,
        raw_path=section.get("train_jsonl", "../output/llm_prep_train.jsonl"),
    )
    out_dir = _resolve_path(config_path=config_path, raw_path=section.get("out_dir", "../output/train"))
    env_file = _resolve_path(config_path=config_path, raw_path=section.get("env_file", "../.env"))

    precision = str(section.get("precision", "bf16")).strip().lower()
    if precision not in _ALLOWED_PRECISIONS:
        raise RuntimeError(
            "Invalid train.precision in config: "
            f"expected one of {sorted(_ALLOWED_PRECISIONS)}, got '{precision}'"
        )

    checkpoint_every_steps = int(section.get("checkpoint_every_steps", 500))
    save_total_limit = int(section.get("save_total_limit", 3))
    max_seq_len = int(section.get("max_seq_len", 1024))
    if checkpoint_every_steps < 1:
        raise RuntimeError("Invalid train.checkpoint_every_steps in config: must be >= 1")
    if save_total_limit < 1:
        raise RuntimeError("Invalid train.save_total_limit in config: must be >= 1")
    if max_seq_len < 64:
        raise RuntimeError("Invalid train.max_seq_len in config: must be >= 64")

    per_device_train_batch_size = int(section.get("per_device_train_batch_size", 1))
    per_device_eval_batch_size = int(section.get("per_device_eval_batch_size", 1))
    gradient_accumulation_steps = int(section.get("gradient_accumulation_steps", 16))
    if per_device_train_batch_size < 1 or per_device_eval_batch_size < 1:
        raise RuntimeError("Invalid train batch sizes in config: values must be >= 1")
    if gradient_accumulation_steps < 1:
        raise RuntimeError("Invalid train.gradient_accumulation_steps in config: must be >= 1")

    learning_rate = float(section.get("learning_rate", 2e-4))
    weight_decay = float(section.get("weight_decay", 0.0))
    warmup_ratio = float(section.get("warmup_ratio", 0.03))
    if learning_rate <= 0:
        raise RuntimeError("Invalid train.learning_rate in config: must be > 0")
    if weight_decay < 0:
        raise RuntimeError("Invalid train.weight_decay in config: must be >= 0")
    if warmup_ratio < 0 or warmup_ratio >= 1:
        raise RuntimeError("Invalid train.warmup_ratio in config: must be >= 0 and < 1")

    dry_run = bool(section.get("dry_run", False))
    dry_run_max_steps = int(section.get("dry_run_max_steps", 20))
    dry_run_sample_rows = int(section.get("dry_run_sample_rows", 2_048))
    dry_run_out_subdir = str(section.get("dry_run_out_subdir", "dry_run")).strip()
    if dry_run_max_steps < 1:
        raise RuntimeError("Invalid train.dry_run_max_steps in config: must be >= 1")
    if dry_run_sample_rows < 128:
        raise RuntimeError("Invalid train.dry_run_sample_rows in config: must be >= 128")
    if not dry_run_out_subdir:
        raise RuntimeError("Invalid train.dry_run_out_subdir in config: must not be empty")

    lora_section = _require_subsection(section=section, section_name="train.lora")
    lora_target_modules = lora_section.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if not isinstance(lora_target_modules, list) or not lora_target_modules:
        raise RuntimeError("Invalid train.lora.target_modules in config: must be non-empty list")
    lora_cfg = LoRAConfig(
        enabled=bool(lora_section.get("enabled", True)),
        r=int(lora_section.get("r", 16)),
        alpha=int(lora_section.get("alpha", 32)),
        dropout=float(lora_section.get("dropout", 0.05)),
        target_modules=[str(v).strip() for v in lora_target_modules if str(v).strip()],
    )
    if lora_cfg.r < 1 or lora_cfg.alpha < 1:
        raise RuntimeError("Invalid train.lora r/alpha in config: values must be >= 1")
    if lora_cfg.dropout < 0 or lora_cfg.dropout >= 1:
        raise RuntimeError("Invalid train.lora.dropout in config: must be >= 0 and < 1")
    if not lora_cfg.target_modules:
        raise RuntimeError("Invalid train.lora.target_modules in config: all values were empty")

    token_section = _require_subsection(section=section, section_name="train.tokens")
    warm_section = _require_subsection(section=section, section_name="train.tokens.warm_start")
    warm_cfg = TokenWarmStartConfig(
        enabled=bool(warm_section.get("enabled", False)),
        semantic_ids_path=_resolve_path(
            config_path=config_path,
            raw_path=warm_section.get(
                "semantic_ids_path", "../data/semantic_ids/conservative/semantic_ids.jsonl"
            ),
        ),
        rqvae_checkpoint_path=_resolve_path(
            config_path=config_path,
            raw_path=warm_section.get(
                "rqvae_checkpoint_path",
                "../../tokenizer/artifacts/rqvae_conservative/rqvae_best_infer.pt",
            ),
        ),
        ridge_lambda=float(warm_section.get("ridge_lambda", 1e-3)),
        max_fit_samples=int(warm_section.get("max_fit_samples", 50_000)),
    )
    if warm_cfg.ridge_lambda < 0:
        raise RuntimeError("Invalid train.tokens.warm_start.ridge_lambda in config: must be >= 0")
    if warm_cfg.max_fit_samples < 128:
        raise RuntimeError("Invalid train.tokens.warm_start.max_fit_samples in config: must be >= 128")

    token_cfg = TokenExtensionConfig(
        semantic_vocab_path=_resolve_path(
            config_path=config_path,
            raw_path=token_section.get(
                "semantic_vocab_path", "../data/semantic_ids/conservative/semantic_vocab.json"
            ),
        ),
        add_special_tokens=bool(token_section.get("add_special_tokens", True)),
        warm_start=warm_cfg,
    )

    phase1 = _build_phase_config(
        section=_require_subsection(section=section, section_name="train.phase1"),
        default_epochs=2,
        default_weights=(0.85, 0.10, 0.05),
    )
    phase2 = _build_phase_config(
        section=_require_subsection(section=section, section_name="train.phase2"),
        default_epochs=3,
        default_weights=(0.10, 0.45, 0.45),
    )

    general_section = _require_subsection(section=section, section_name="train.general_mix")
    raw_sources = general_section.get("sources", ["SlimOrca", "OpenHermes"])
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError("Invalid train.general_mix.sources in config: must be non-empty list")
    sources = [str(v).strip() for v in raw_sources if str(v).strip()]
    if not sources:
        raise RuntimeError("Invalid train.general_mix.sources in config: all values were empty")
    invalid_sources = sorted(set(sources) - _ALLOWED_GENERAL_SOURCES)
    if invalid_sources:
        raise RuntimeError(
            "Invalid train.general_mix.sources in config: "
            f"unsupported values {invalid_sources}; allowed={sorted(_ALLOWED_GENERAL_SOURCES)}"
        )
    general_cfg = GeneralMixConfig(
        ratio=float(general_section.get("ratio", 0.15)),
        sources=sources,
        max_rows_per_source=int(general_section.get("max_rows_per_source", 60_000)),
        cache_dir=_resolve_path(config_path=config_path, raw_path=general_section.get("cache_dir", "../output/hf_cache")),
        seed=int(general_section.get("seed", 42)),
    )
    if general_cfg.ratio < 0 or general_cfg.ratio >= 1:
        raise RuntimeError("Invalid train.general_mix.ratio in config: must be >= 0 and < 1")
    if general_cfg.max_rows_per_source < 0:
        raise RuntimeError("Invalid train.general_mix.max_rows_per_source in config: must be >= 0")

    eval_section = _require_subsection(section=section, section_name="train.eval")
    eval_cfg = EvalConfig(
        english_eval_dataset=str(eval_section.get("english_eval_dataset", "wikitext2")).strip(),
        eval_every_steps=int(eval_section.get("eval_every_steps", 500)),
        eval_max_samples=int(eval_section.get("eval_max_samples", 512)),
        eval_batch_size=int(eval_section.get("eval_batch_size", 2)),
    )
    if eval_cfg.eval_every_steps < 1:
        raise RuntimeError("Invalid train.eval.eval_every_steps in config: must be >= 1")
    if eval_cfg.eval_max_samples < 1:
        raise RuntimeError("Invalid train.eval.eval_max_samples in config: must be >= 1")
    if eval_cfg.eval_batch_size < 1:
        raise RuntimeError("Invalid train.eval.eval_batch_size in config: must be >= 1")

    wandb_section = _require_subsection(section=section, section_name="train.wandb")
    raw_tags = wandb_section.get("tags", [])
    if not isinstance(raw_tags, list):
        raise RuntimeError("Invalid train.wandb.tags in config: must be list")
    wandb_mode = str(wandb_section.get("mode", "offline")).strip().lower()
    if wandb_mode not in _ALLOWED_WANDB_MODES:
        raise RuntimeError(
            "Invalid train.wandb.mode in config: "
            f"expected one of {sorted(_ALLOWED_WANDB_MODES)}, got '{wandb_mode}'"
        )
    wandb_cfg = WandbConfig(
        enabled=bool(wandb_section.get("enabled", False)),
        mode=wandb_mode,
        project=str(wandb_section.get("project", "")).strip(),
        entity=str(wandb_section.get("entity", "")).strip(),
        api_key=str(wandb_section.get("api_key", "")).strip(),
        run_name=str(wandb_section.get("run_name", "")).strip(),
        group=str(wandb_section.get("group", "")).strip(),
        tags=[str(v).strip() for v in raw_tags if str(v).strip()],
        project_env=str(wandb_section.get("project_env", "WANDB_PROJECT")).strip(),
        entity_env=str(wandb_section.get("entity_env", "WANDB_ENTITY")).strip(),
        api_key_env=str(wandb_section.get("api_key_env", "WANDB_API_KEY")).strip(),
    )

    return TrainConfig(
        base_model=base_model,
        dry_run_model_name=str(section.get("dry_run_model_name", "sshleifer/tiny-gpt2")).strip(),
        train_jsonl=train_jsonl,
        out_dir=out_dir,
        env_file=env_file,
        rebuild=bool(section.get("rebuild", True)),
        seed=int(section.get("seed", 42)),
        device=str(section.get("device", "auto")).strip(),
        precision=precision,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=bool(section.get("gradient_checkpointing", True)),
        max_grad_norm=float(section.get("max_grad_norm", 1.0)),
        checkpoint_every_steps=checkpoint_every_steps,
        save_total_limit=save_total_limit,
        resume_from=str(section.get("resume_from", "")).strip(),
        hf_token=str(section.get("hf_token", "")).strip(),
        hf_token_env=str(section.get("hf_token_env", "HF_TOKEN")).strip() or "HF_TOKEN",
        dry_run=dry_run,
        dry_run_max_steps=dry_run_max_steps,
        dry_run_sample_rows=dry_run_sample_rows,
        dry_run_out_subdir=dry_run_out_subdir,
        lora=lora_cfg,
        tokens=token_cfg,
        phase1=phase1,
        phase2=phase2,
        general_mix=general_cfg,
        eval=eval_cfg,
        wandb=wandb_cfg,
    )


def _require_subsection(section: dict[str, Any], section_name: str) -> dict[str, Any]:
    name_parts = section_name.split(".")
    current: Any = section
    for part in name_parts[1:]:
        if not isinstance(current, dict):
            raise RuntimeError(f"Missing required config section: [{section_name}]")
        current = current.get(part)
    if not isinstance(current, dict):
        raise RuntimeError(f"Missing required config section: [{section_name}]")
    return current


def _build_phase_config(
    *,
    section: dict[str, Any],
    default_epochs: int,
    default_weights: tuple[float, float, float],
) -> PhaseConfig:
    cfg = PhaseConfig(
        epochs=int(section.get("epochs", default_epochs)),
        weight_task_a=float(section.get("weight_task_a", default_weights[0])),
        weight_task_b=float(section.get("weight_task_b", default_weights[1])),
        weight_task_c=float(section.get("weight_task_c", default_weights[2])),
        max_domain_rows=int(section.get("max_domain_rows", 0)),
    )
    if cfg.epochs < 1:
        raise RuntimeError("Invalid phase epochs in config: must be >= 1")
    if cfg.max_domain_rows < 0:
        raise RuntimeError("Invalid phase max_domain_rows in config: must be >= 0")
    if any(weight <= 0 for weight in (cfg.weight_task_a, cfg.weight_task_b, cfg.weight_task_c)):
        raise RuntimeError("Invalid phase task weights in config: each weight must be > 0")
    total = cfg.weight_task_a + cfg.weight_task_b + cfg.weight_task_c
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(f"Invalid phase task weights in config: must sum to 1.0, got {total:.6f}")
    return cfg
