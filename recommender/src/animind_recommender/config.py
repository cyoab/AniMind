from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .prep import PrepConfig

_ALLOWED_MASK_MODES = {"last", "random"}


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
