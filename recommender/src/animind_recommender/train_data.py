from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from .train_types import GeneralMixConfig, PhaseConfig

_KNOWN_GENERAL_SOURCES = {"SlimOrca", "OpenHermes"}


def load_domain_rows(*, path: Path, limit: int = 0) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise RuntimeError(f"Training dataset file not found: {path}")
    by_task: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            task = str(payload.get("task", "")).strip()
            if task not in by_task:
                raise RuntimeError(f"Unsupported task '{task}' in {path}:{line_number}")
            input_text = str(payload.get("input", "")).strip()
            output_text = str(payload.get("output", "")).strip()
            if not input_text or not output_text:
                continue
            by_task[task].append(
                {
                    "task": task,
                    "input": input_text,
                    "output": output_text,
                    "source": "animind_domain",
                }
            )
            if limit > 0 and sum(len(rows) for rows in by_task.values()) >= limit:
                break
    total = sum(len(rows) for rows in by_task.values())
    if total == 0:
        raise RuntimeError(f"No usable domain rows found in {path}")
    return by_task


def load_general_instruction_rows(
    *,
    config: GeneralMixConfig,
    console: Console,
) -> list[dict[str, Any]]:
    if config.ratio <= 0:
        return []

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "General mix ratio is > 0 but datasets package is missing. Install with: uv add datasets"
        ) from exc

    out_rows: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    for source_name in config.sources:
        source = str(source_name).strip()
        if source not in _KNOWN_GENERAL_SOURCES:
            raise RuntimeError(
                f"Unsupported general mix source '{source}'. Allowed={sorted(_KNOWN_GENERAL_SOURCES)}"
            )
        dataset_id, split = _resolve_source_dataset(source)
        console.log(f"Loading general source {source}: {dataset_id} [{split}]")
        dataset = load_dataset(dataset_id, split=split, cache_dir=str(config.cache_dir))
        if config.max_rows_per_source > 0:
            sample_cap = min(config.max_rows_per_source, len(dataset))
            if sample_cap < len(dataset):
                dataset = dataset.shuffle(seed=config.seed).select(range(sample_cap))
        for row in dataset:
            converted = _convert_general_row(row=row, source=source)
            if converted is not None:
                out_rows.append(converted)

    rng.shuffle(out_rows)
    return out_rows


def build_stage_examples(
    *,
    phase_name: str,
    phase_cfg: PhaseConfig,
    domain_rows: dict[str, list[dict[str, Any]]],
    general_rows: list[dict[str, Any]],
    general_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    total_available = sum(len(domain_rows[key]) for key in ("A", "B", "C"))
    if total_available <= 0:
        raise RuntimeError("No domain rows available to build stage dataset.")

    domain_target = total_available if phase_cfg.max_domain_rows <= 0 else min(total_available, phase_cfg.max_domain_rows)
    if domain_target < 128:
        raise RuntimeError(f"Stage {phase_name} domain rows are too small: {domain_target}")

    weight_map = {
        "A": phase_cfg.weight_task_a,
        "B": phase_cfg.weight_task_b,
        "C": phase_cfg.weight_task_c,
    }
    stage_domain_rows: list[dict[str, Any]] = []
    for task in ("A", "B", "C"):
        quota = int(round(domain_target * weight_map[task]))
        if quota <= 0:
            continue
        stage_domain_rows.extend(_sample_rows(rows=domain_rows[task], count=quota, rng=rng))

    if not stage_domain_rows:
        raise RuntimeError(f"Stage {phase_name} has no domain rows after weighting.")

    rng.shuffle(stage_domain_rows)
    stage_rows = list(stage_domain_rows)
    if general_ratio > 0:
        if not general_rows:
            raise RuntimeError(
                f"Stage {phase_name} requested general_ratio={general_ratio} but no general rows are available."
            )
        general_target = int(round(len(stage_domain_rows) * general_ratio / max(1e-9, 1.0 - general_ratio)))
        stage_rows.extend(_sample_rows(rows=general_rows, count=general_target, rng=rng))
    rng.shuffle(stage_rows)
    return stage_rows


def split_train_eval(
    *,
    rows: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise RuntimeError("Cannot split empty dataset.")
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise RuntimeError("eval_ratio must be > 0 and < 1.")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    eval_size = max(1, int(round(len(rows) * eval_ratio)))
    if eval_size >= len(rows):
        eval_size = max(1, len(rows) // 5)
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    if not train_rows:
        raise RuntimeError("Split produced empty train_rows; adjust eval_ratio or dataset size.")
    return train_rows, eval_rows


def summarize_task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("task", "unknown"))] += 1
    return dict(sorted(counts.items()))


def format_instruction_pair(*, input_text: str, output_text: str) -> str:
    return (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\n"
        f"{input_text}\n"
        "<|assistant|>\n"
        f"{output_text}"
    )


def _sample_rows(*, rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not rows:
        return []
    if len(rows) >= count:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [dict(rows[i]) for i in indices[:count]]
    sampled: list[dict[str, Any]] = []
    while len(sampled) < count:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for index in indices:
            sampled.append(dict(rows[index]))
            if len(sampled) >= count:
                break
    return sampled


def _resolve_source_dataset(source: str) -> tuple[str, str]:
    if source == "SlimOrca":
        return "Open-Orca/SlimOrca-Dedup", "train"
    if source == "OpenHermes":
        return "teknium/OpenHermes-2.5", "train"
    raise RuntimeError(f"Unsupported source: {source}")


def _convert_general_row(*, row: dict[str, Any], source: str) -> dict[str, Any] | None:
    parsed = _extract_prompt_response(row)
    if parsed is None:
        return None
    prompt, response = parsed
    if not prompt or not response:
        return None
    return {
        "task": "GEN",
        "input": prompt,
        "output": response,
        "source": source,
    }


def _extract_prompt_response(row: dict[str, Any]) -> tuple[str, str] | None:
    if "instruction" in row and "output" in row:
        prompt = " ".join(str(row.get("instruction", "")).split())
        extra = " ".join(str(row.get("input", "")).split())
        if extra:
            prompt = f"{prompt}\n{extra}".strip()
        response = " ".join(str(row.get("output", "")).split())
        if prompt and response:
            return prompt, response

    if "prompt" in row and "response" in row:
        prompt = " ".join(str(row.get("prompt", "")).split())
        response = " ".join(str(row.get("response", "")).split())
        if prompt and response:
            return prompt, response

    if "conversations" in row and isinstance(row["conversations"], list):
        prompt, response = _extract_from_conversations(row["conversations"])
        if prompt and response:
            return prompt, response

    if "messages" in row and isinstance(row["messages"], list):
        prompt, response = _extract_from_messages(row["messages"])
        if prompt and response:
            return prompt, response

    return None


def _extract_from_conversations(conversations: list[Any]) -> tuple[str, str]:
    user_value = ""
    assistant_value = ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", "") or item.get("role", "")).strip().lower()
        value = str(item.get("value", "") or item.get("content", "")).strip()
        if role in {"human", "user"} and not user_value:
            user_value = " ".join(value.split())
        if role in {"gpt", "assistant"} and not assistant_value:
            assistant_value = " ".join(value.split())
        if user_value and assistant_value:
            break
    return user_value, assistant_value


def _extract_from_messages(messages: list[Any]) -> tuple[str, str]:
    user_value = ""
    assistant_value = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        value = str(item.get("content", "")).strip()
        if role == "user" and not user_value:
            user_value = " ".join(value.split())
        if role == "assistant" and not assistant_value:
            assistant_value = " ".join(value.split())
        if user_value and assistant_value:
            break
    return user_value, assistant_value
