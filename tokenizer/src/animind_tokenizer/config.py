from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .embed import EmbedConfig
from .prep import PrepConfig


def build_prep_config(config_path: Path) -> PrepConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="prep")

    source_db = _resolve_path(config_path=config_path, raw_path=section.get("source_db", "../output/anilist.sqlite"))
    out_dir = _resolve_path(config_path=config_path, raw_path=section.get("out_dir", "./output"))
    rebuild = bool(section.get("rebuild", True))
    export_parquet = bool(section.get("export_parquet", True))
    limit = int(section.get("limit", 0))
    if limit < 0:
        raise RuntimeError("Invalid prep.limit in config: must be >= 0")

    return PrepConfig(
        source_db=source_db,
        out_dir=out_dir,
        rebuild=rebuild,
        export_parquet=export_parquet,
        limit=limit,
    )


def build_embed_config(config_path: Path) -> EmbedConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="embedd")

    tokenizer_db = _resolve_path(
        config_path=config_path,
        raw_path=section.get("tokenizer_db", "./output/tokenizer.sqlite"),
    )
    rebuild = bool(section.get("rebuild", True))
    limit = int(section.get("limit", 0))
    if limit < 0:
        raise RuntimeError("Invalid embedd.limit in config: must be >= 0")

    batch_size = int(section.get("batch_size", 8))
    max_length = int(section.get("max_length", 2048))
    if batch_size < 1:
        raise RuntimeError("Invalid embedd.batch_size in config: must be >= 1")
    if max_length < 8:
        raise RuntimeError("Invalid embedd.max_length in config: must be >= 8")

    return EmbedConfig(
        tokenizer_db=tokenizer_db,
        rebuild=rebuild,
        limit=limit,
        model_name=str(section.get("model_name", "tencent/KaLM-Embedding-Gemma3-12B-2511")),
        batch_size=batch_size,
        max_length=max_length,
        device=str(section.get("device", "auto")),
        normalize=bool(section.get("normalize", True)),
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

