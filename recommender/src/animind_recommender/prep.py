from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

PREP_PHASE = Literal["prep"]
DEFAULT_ALLOWED_STATUSES = [1, 2, 3, 4, 6]
UNKNOWN_TEXT = "Unknown"
SPECIAL_TOKEN_ANIME_START = "<anime_start>"
SPECIAL_TOKEN_ANIME_END = "<anime_end>"
SENTIMENT_LIKED = "<liked>"
SENTIMENT_DISLIKED = "<disliked>"
SENTIMENT_DROPPED = "<dropped>"
SENTIMENT_WATCH = "<watch>"


@dataclass(slots=True)
class PrepConfig:
    source_db: Path = Path("../../output/anilist.sqlite")
    semantic_ids_path: Path = Path("../data/semantic_ids/conservative/semantic_ids.jsonl")
    out_dir: Path = Path("../output")
    rebuild: bool = True
    seed: int = 42
    target_examples: int = 400_000
    split_task_a: float = 0.50
    split_task_b: float = 0.25
    split_task_c: float = 0.25
    max_task_a_templates_per_anime: int = 12
    task_b_min_history: int = 4
    task_b_max_history: int = 30
    task_b_mask_modes: list[str] = field(default_factory=lambda: ["last", "random"])
    task_b_positive_score_min: int = 7
    task_b_allowed_statuses: list[int] = field(default_factory=lambda: list(DEFAULT_ALLOWED_STATUSES))
    export_parquet: bool = False
    write_manifest: bool = True

    @property
    def output_jsonl(self) -> Path:
        return self.out_dir / "llm_prep_train.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.out_dir / "llm_prep_summary.json"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "llm_prep_manifest.json"

    @property
    def output_parquet(self) -> Path:
        return self.out_dir / "llm_prep_train.parquet"


@dataclass(slots=True)
class SemanticAnime:
    anime_id: int
    name: str
    english_name: str
    genres: str
    semantic_id: str
    tokens: list[str]
    codes: list[int]


@dataclass(slots=True)
class AnimeMeta:
    anime_id: int
    name: str
    english_name: str
    genres: str
    synopsis: str
    studios: str
    year: int | None
    premiered: str
    anime_type: str
    source: str

    def preferred_title(self) -> str:
        if self.english_name:
            return self.english_name
        if self.name:
            return self.name
        return UNKNOWN_TEXT


@dataclass(slots=True)
class TaskQuotas:
    task_a: int
    task_b: int
    task_c: int


def run_pipeline(config: PrepConfig, phase: PREP_PHASE = "prep") -> None:
    if phase != "prep":
        raise ValueError(f"Unsupported phase: {phase}")
    run_prep(config=config)


def run_prep(config: PrepConfig) -> None:
    console = Console()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if not config.source_db.exists():
        raise RuntimeError(f"Source database not found: {config.source_db}")
    if not config.semantic_ids_path.exists():
        raise RuntimeError(f"Semantic IDs file not found: {config.semantic_ids_path}")
    if not config.rebuild and config.output_jsonl.exists() and config.output_jsonl.stat().st_size > 0:
        console.log(
            f"[yellow]Reuse enabled:[/yellow] keeping existing dataset {config.output_jsonl}."
        )
        return

    generated_at = _now()
    quotas = _compute_quotas(config=config)
    rng = random.Random(config.seed)
    console.rule("[bold cyan]Recommender Prep Dataset[/bold cyan]")
    console.log(
        "Starting prep run with "
        f"target_examples={config.target_examples:,}, "
        f"quota(A/B/C)={quotas.task_a:,}/{quotas.task_b:,}/{quotas.task_c:,}, "
        f"seed={config.seed}."
    )

    stage_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with Progress(*stage_columns, console=console) as stage_progress:
        stage_task = stage_progress.add_task("[cyan]Prep stages[/cyan]", total=8)

        stage_progress.update(stage_task, description="[cyan]1/8 Load semantic ID map[/cyan]")
        semantic_map = _load_semantic_map(config.semantic_ids_path)
        stage_progress.advance(stage_task)
        console.log(f"Loaded semantic map rows: {len(semantic_map):,}")

        stage_progress.update(stage_task, description="[cyan]2/8 Load anime metadata[/cyan]")
        anime_meta = _load_anime_meta(config.source_db, required_ids=set(semantic_map.keys()))
        if not anime_meta:
            raise RuntimeError("No anime metadata rows could be loaded from source database.")
        stage_progress.advance(stage_task)
        console.log(f"Loaded anime metadata rows: {len(anime_meta):,}")

        stage_progress.update(stage_task, description="[cyan]3/8 Build Task A pool[/cyan]")
        task_a_pool, task_a_stats = _build_task_a_pool(
            semantic_map=semantic_map,
            anime_meta=anime_meta,
            max_templates_per_anime=config.max_task_a_templates_per_anime,
        )
        stage_progress.advance(stage_task)
        console.log(f"Task A pool size: {len(task_a_pool):,}")

        stage_progress.update(stage_task, description="[cyan]4/8 Build Task B/C seed pools[/cyan]")
        task_b_pool, task_c_seed_pool, stream_stats = _build_task_b_and_task_c_seeds(
            config=config,
            semantic_map=semantic_map,
            desired_task_b_pool=max(quotas.task_b * 2, 1),
            desired_task_c_seed_pool=max(quotas.task_c * 2, 1),
            console=console,
        )
        stage_progress.advance(stage_task)
        console.log(
            "Task B/C seed pools: "
            f"task_b_pool={len(task_b_pool):,}, task_c_seed_pool={len(task_c_seed_pool):,}."
        )

        stage_progress.update(stage_task, description="[cyan]5/8 Build Task C pool[/cyan]")
        task_c_pool, task_c_stats = _build_task_c_pool(
            task_c_seed_pool=task_c_seed_pool,
            semantic_map=semantic_map,
            anime_meta=anime_meta,
        )
        stage_progress.advance(stage_task)
        console.log(f"Task C pool size: {len(task_c_pool):,}")

        stage_progress.update(stage_task, description="[cyan]6/8 Sample quotas and shuffle[/cyan]")
        task_a_rows = _sample_to_quota(task_a_pool, quotas.task_a, rng=rng)
        task_b_rows = _sample_to_quota(task_b_pool, quotas.task_b, rng=rng)
        task_c_rows = _sample_to_quota(task_c_pool, quotas.task_c, rng=rng)

        all_rows = [*task_a_rows, *task_b_rows, *task_c_rows]
        if len(all_rows) != config.target_examples:
            raise RuntimeError(
                "Generated row count does not match target_examples: "
                f"got {len(all_rows)} expected {config.target_examples}"
            )

        rng.shuffle(all_rows)
        for index, row in enumerate(all_rows, start=1):
            row["example_id"] = f"prep_{index:09d}"
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]7/8 Write training artifacts[/cyan]")
        _write_jsonl(config.output_jsonl, all_rows)
        if config.export_parquet:
            _write_parquet(config.output_parquet, all_rows)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]8/8 Write summary and manifest[/cyan]")
        per_task_counts = Counter(row["task"] for row in all_rows)
        summary = {
            "generated_at": generated_at,
            "target_examples": config.target_examples,
            "counts": {
                "task_a": int(per_task_counts.get("A", 0)),
                "task_b": int(per_task_counts.get("B", 0)),
                "task_c": int(per_task_counts.get("C", 0)),
            },
            "quotas": asdict(quotas),
            "pool_sizes": {
                "task_a_pool": len(task_a_pool),
                "task_b_pool": len(task_b_pool),
                "task_c_pool": len(task_c_pool),
                "task_c_seed_pool": len(task_c_seed_pool),
            },
            "task_a_stats": task_a_stats,
            "stream_stats": stream_stats,
            "task_c_stats": task_c_stats,
            "semantic_rows": len(semantic_map),
            "anime_meta_rows": len(anime_meta),
        }
        _write_json(config.summary_json, summary)

        if config.write_manifest:
            manifest = {
                "generated_at": generated_at,
                "output_jsonl": str(config.output_jsonl),
                "output_parquet": str(config.output_parquet) if config.export_parquet else None,
                "summary_json": str(config.summary_json),
                "source_db": str(config.source_db),
                "semantic_ids_path": str(config.semantic_ids_path),
                "config": {
                    **asdict(config),
                    "source_db": str(config.source_db),
                    "semantic_ids_path": str(config.semantic_ids_path),
                    "out_dir": str(config.out_dir),
                },
            }
            _write_json(config.manifest_json, manifest)
        stage_progress.advance(stage_task)

    console.log(
        "Prep complete: "
        f"rows={config.target_examples:,}, "
        f"task_a={per_task_counts.get('A', 0):,}, "
        f"task_b={per_task_counts.get('B', 0):,}, "
        f"task_c={per_task_counts.get('C', 0):,}, "
        f"out={config.output_jsonl}."
    )


def _compute_quotas(config: PrepConfig) -> TaskQuotas:
    task_a = int(round(config.target_examples * config.split_task_a))
    task_b = int(round(config.target_examples * config.split_task_b))
    task_c = config.target_examples - task_a - task_b
    if task_a < 0 or task_b < 0 or task_c < 0:
        raise RuntimeError("Invalid quota computation. Check split values and target_examples.")
    return TaskQuotas(task_a=task_a, task_b=task_b, task_c=task_c)


def _load_semantic_map(path: Path) -> dict[int, SemanticAnime]:
    semantic_map: dict[int, SemanticAnime] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            anime_id = int(payload["anime_id"])
            semantic_id = str(payload["semantic_id"]).strip()
            if not semantic_id:
                raise RuntimeError(f"Empty semantic_id for anime_id={anime_id} in {path}:{line_number}")
            semantic_map[anime_id] = SemanticAnime(
                anime_id=anime_id,
                name=_clean_text(payload.get("name")),
                english_name=_clean_text(payload.get("english_name")),
                genres=_clean_text(payload.get("genres")),
                semantic_id=semantic_id,
                tokens=[str(token) for token in payload.get("tokens", [])],
                codes=[int(code) for code in payload.get("codes", [])],
            )
    if not semantic_map:
        raise RuntimeError(f"No semantic rows loaded from {path}")
    return semantic_map


def _load_anime_meta(source_db: Path, required_ids: set[int]) -> dict[int, AnimeMeta]:
    uri = f"file:{source_db.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        if not _table_exists(conn, "anime"):
            raise RuntimeError(f"Source database missing required table: anime ({source_db})")
        cursor = conn.execute(
            """
            SELECT
                anime_id,
                name,
                english_name,
                genres,
                synopsis,
                studios,
                year,
                premiered,
                type,
                source
            FROM anime
            ORDER BY anime_id
            """
        )
        meta: dict[int, AnimeMeta] = {}
        for row in cursor:
            anime_id = int(row[0])
            if anime_id not in required_ids:
                continue
            meta[anime_id] = AnimeMeta(
                anime_id=anime_id,
                name=_clean_text(row[1]),
                english_name=_clean_text(row[2]),
                genres=_clean_text(row[3]),
                synopsis=_clean_text(row[4]),
                studios=_clean_text(row[5]),
                year=_coerce_nonneg_int(row[6]),
                premiered=_clean_text(row[7]),
                anime_type=_clean_text(row[8]),
                source=_clean_text(row[9]),
            )
    return meta


def _build_task_a_pool(
    *,
    semantic_map: dict[int, SemanticAnime],
    anime_meta: dict[int, AnimeMeta],
    max_templates_per_anime: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    template_counter: Counter[str] = Counter()
    skipped_without_meta = 0

    for anime_id in sorted(semantic_map.keys()):
        semantic = semantic_map[anime_id]
        meta = anime_meta.get(anime_id)
        if meta is None:
            skipped_without_meta += 1
            continue
        title = meta.preferred_title()
        genres = meta.genres or semantic.genres
        studios = meta.studios
        year_text = str(meta.year) if meta.year is not None else (meta.premiered or "")
        synopsis = meta.synopsis
        themes = _themes_from_genres(genres)

        templates: list[dict[str, Any]] = [
            _make_task_a_row(
                template_id="sid_to_title",
                anime_id=anime_id,
                prompt=f"The anime {semantic.semantic_id} has the title",
                completion=title,
            ),
            _make_task_a_row(
                template_id="title_to_sid",
                anime_id=anime_id,
                prompt=f"The anime titled '{title}' has semantic id",
                completion=semantic.semantic_id,
            ),
        ]
        if genres:
            templates.extend(
                [
                    _make_task_a_row(
                        template_id="sid_to_genres",
                        anime_id=anime_id,
                        prompt=f"The anime {semantic.semantic_id} belongs to genres",
                        completion=genres,
                    ),
                    _make_task_a_row(
                        template_id="title_to_genres",
                        anime_id=anime_id,
                        prompt=f"The anime titled '{title}' belongs to genres",
                        completion=genres,
                    ),
                ]
            )
        if studios:
            templates.extend(
                [
                    _make_task_a_row(
                        template_id="sid_to_studio",
                        anime_id=anime_id,
                        prompt=f"The anime {semantic.semantic_id} is produced by studio",
                        completion=studios,
                    ),
                    _make_task_a_row(
                        template_id="title_studio_to_sid",
                        anime_id=anime_id,
                        prompt=f"The anime titled '{title}' by studio {studios} has semantic id",
                        completion=semantic.semantic_id,
                    ),
                ]
            )
        if year_text:
            templates.extend(
                [
                    _make_task_a_row(
                        template_id="sid_to_year",
                        anime_id=anime_id,
                        prompt=f"The anime {semantic.semantic_id} released in year",
                        completion=year_text,
                    ),
                    _make_task_a_row(
                        template_id="title_year_to_sid",
                        anime_id=anime_id,
                        prompt=f"The anime titled '{title}' released in {year_text} has semantic id",
                        completion=semantic.semantic_id,
                    ),
                ]
            )
        if synopsis:
            templates.append(
                _make_task_a_row(
                    template_id="sid_to_synopsis",
                    anime_id=anime_id,
                    prompt=f"The anime {semantic.semantic_id} synopsis:",
                    completion=synopsis,
                )
            )
        if themes:
            templates.append(
                _make_task_a_row(
                    template_id="sid_to_themes",
                    anime_id=anime_id,
                    prompt=f"The anime {semantic.semantic_id} has notable themes",
                    completion=themes,
                )
            )

        for row in templates[:max_templates_per_anime]:
            rows.append(row)
            template_counter[row["meta"]["template_id"]] += 1

    stats = {
        "rows_generated": len(rows),
        "anime_skipped_without_meta": skipped_without_meta,
        "template_counts": dict(sorted(template_counter.items())),
    }
    return rows, stats


def _make_task_a_row(*, template_id: str, anime_id: int, prompt: str, completion: str) -> dict[str, Any]:
    return {
        "task": "A",
        "input": _clean_text(prompt),
        "output": _clean_text(completion),
        "anime_ids": [int(anime_id)],
        "user_id": None,
        "meta": {"template_id": template_id},
    }


def _build_task_b_and_task_c_seeds(
    *,
    config: PrepConfig,
    semantic_map: dict[int, SemanticAnime],
    desired_task_b_pool: int,
    desired_task_c_seed_pool: int,
    console: Console | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    allowed_statuses = sorted(set(int(v) for v in config.task_b_allowed_statuses))
    placeholders = ",".join("?" for _ in allowed_statuses)
    uri = f"file:{config.source_db.resolve()}?mode=ro&immutable=1"

    task_b_rows: list[dict[str, Any]] = []
    task_c_seeds: list[dict[str, Any]] = []
    rows_scanned = 0
    users_scanned = 0
    users_with_valid_history = 0
    users_with_task_c_seeds = 0
    dropped_unknown_anime = 0
    dropped_short_history = 0
    rng = random.Random(config.seed)

    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        if not _table_exists(conn, "user_anime"):
            raise RuntimeError(f"Source database missing required table: user_anime ({config.source_db})")
        query = f"""
            SELECT user_id, anime_id, score, watching_status, watched_episodes, scraped_at
            FROM user_anime
            WHERE watching_status IN ({placeholders})
            ORDER BY user_id, anime_id
        """
        cursor = conn.execute(query, tuple(allowed_statuses))

        current_user_id: int | None = None
        bucket: list[tuple[int, int, int, int, str]] = []

        def flush_user(user_id: int, user_rows: list[tuple[int, int, int, int, str]]) -> None:
            nonlocal users_scanned
            nonlocal users_with_valid_history
            nonlocal users_with_task_c_seeds
            nonlocal dropped_short_history
            nonlocal dropped_unknown_anime
            users_scanned += 1
            filtered: list[tuple[int, int, int, int, str]] = []
            for anime_id, score, status, watched_episodes, scraped_at in user_rows:
                if anime_id not in semantic_map:
                    dropped_unknown_anime += 1
                    continue
                filtered.append((anime_id, score, status, watched_episodes, scraped_at))
            if len(filtered) < config.task_b_min_history:
                dropped_short_history += 1
            else:
                users_with_valid_history += 1
                filtered.sort(
                    key=lambda item: (
                        item[4],
                        item[2],
                        -item[1],
                        -item[3],
                        item[0],
                    )
                )
                if len(filtered) > config.task_b_max_history:
                    filtered = filtered[-config.task_b_max_history :]
                mask_indices = _pick_mask_indices(
                    history_len=len(filtered),
                    modes=config.task_b_mask_modes,
                    rng=rng,
                )
                for mode, mask_index in mask_indices:
                    prompt = _build_task_b_prompt(
                        history=filtered,
                        mask_index=mask_index,
                        semantic_map=semantic_map,
                        positive_score_min=config.task_b_positive_score_min,
                    )
                    target_sid = semantic_map[filtered[mask_index][0]].semantic_id
                    task_b_rows.append(
                        {
                            "task": "B",
                            "input": prompt,
                            "output": target_sid,
                            "anime_ids": [int(item[0]) for item in filtered],
                            "user_id": int(user_id),
                            "meta": {
                                "template_id": "watch_sequence_mask",
                                "mask_mode": mode,
                                "mask_index": int(mask_index),
                                "history_len": int(len(filtered)),
                            },
                        }
                    )

            liked_sorted = sorted(
                [
                    (anime_id, score)
                    for anime_id, score, _status, _watched_episodes, _scraped_at in filtered
                    if score >= config.task_b_positive_score_min
                ],
                key=lambda item: (-item[1], item[0]),
            )
            liked_ids = [anime_id for anime_id, _score in liked_sorted]
            if liked_ids:
                users_with_task_c_seeds += 1
                task_c_seeds.append(
                    {"kind": "single", "anime_ids": [int(liked_ids[0])], "user_id": int(user_id)}
                )
            if len(liked_ids) >= 3:
                task_c_seeds.append(
                    {
                        "kind": "triple",
                        "anime_ids": [int(liked_ids[0]), int(liked_ids[1]), int(liked_ids[2])],
                        "user_id": int(user_id),
                    }
                )
            if console is not None and users_scanned > 0 and users_scanned % 25_000 == 0:
                console.log(
                    "Task B/C scan progress: "
                    f"users={users_scanned:,}, "
                    f"rows={rows_scanned:,}, "
                    f"task_b_pool={len(task_b_rows):,}, "
                    f"task_c_seed_pool={len(task_c_seeds):,}."
                )

        for user_id_raw, anime_id_raw, score_raw, status_raw, watched_raw, scraped_at_raw in cursor:
            rows_scanned += 1
            user_id = int(user_id_raw)
            anime_id = int(anime_id_raw)
            score = int(score_raw)
            status = int(status_raw)
            watched = int(watched_raw)
            scraped_at = _clean_text(scraped_at_raw)
            if current_user_id is None:
                current_user_id = user_id
            if user_id != current_user_id:
                flush_user(current_user_id, bucket)
                bucket = []
                current_user_id = user_id
                if len(task_b_rows) >= desired_task_b_pool and len(task_c_seeds) >= desired_task_c_seed_pool:
                    break
            bucket.append((anime_id, score, status, watched, scraped_at))

        if current_user_id is not None and bucket:
            flush_user(current_user_id, bucket)

    stream_stats = {
        "rows_scanned": rows_scanned,
        "users_scanned": users_scanned,
        "users_with_valid_history": users_with_valid_history,
        "users_with_task_c_seeds": users_with_task_c_seeds,
        "dropped_unknown_anime": dropped_unknown_anime,
        "dropped_short_history_users": dropped_short_history,
        "task_b_rows_generated": len(task_b_rows),
        "task_c_seed_rows_generated": len(task_c_seeds),
    }
    return task_b_rows, task_c_seeds, stream_stats


def _build_task_b_prompt(
    *,
    history: list[tuple[int, int, int, int, str]],
    mask_index: int,
    semantic_map: dict[int, SemanticAnime],
    positive_score_min: int,
) -> str:
    chunks: list[str] = []
    for index, (anime_id, score, status, _watched, _scraped_at) in enumerate(history):
        semantic_id = semantic_map[anime_id].semantic_id
        visible_sid = "[MASK]" if index == mask_index else semantic_id
        verb = "A user watched" if index == 0 else "then watched"
        segment = f"{verb} {SPECIAL_TOKEN_ANIME_START}{visible_sid}{SPECIAL_TOKEN_ANIME_END}"
        if index != mask_index:
            sentiment = _score_to_sentiment(score=score, status=status, positive_score_min=positive_score_min)
            segment = f"{segment} {sentiment}"
        chunks.append(segment)
    return " ".join(chunks)


def _pick_mask_indices(
    *,
    history_len: int,
    modes: list[str],
    rng: random.Random,
) -> list[tuple[str, int]]:
    if history_len < 2:
        return []
    indices: list[tuple[str, int]] = []
    seen: set[int] = set()
    for mode in modes:
        mask_index: int
        if mode == "last":
            mask_index = history_len - 1
        elif mode == "random":
            upper_bound = max(1, history_len - 1)
            mask_index = int(rng.randrange(0, upper_bound))
        else:
            continue
        if mask_index in seen:
            continue
        seen.add(mask_index)
        indices.append((mode, mask_index))
    return indices


def _score_to_sentiment(*, score: int, status: int, positive_score_min: int) -> str:
    if status == 4:
        return SENTIMENT_DROPPED
    if score >= positive_score_min:
        return SENTIMENT_LIKED
    if 0 < score <= max(1, positive_score_min // 2):
        return SENTIMENT_DISLIKED
    return SENTIMENT_WATCH


def _build_task_c_pool(
    *,
    task_c_seed_pool: list[dict[str, Any]],
    semantic_map: dict[int, SemanticAnime],
    anime_meta: dict[int, AnimeMeta],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    skipped_missing_ids = 0
    for seed in task_c_seed_pool:
        kind = str(seed["kind"])
        anime_ids = [int(v) for v in seed["anime_ids"]]
        if any(anime_id not in semantic_map for anime_id in anime_ids):
            skipped_missing_ids += 1
            continue
        user_id = int(seed["user_id"])
        if kind == "triple":
            sid_values = [semantic_map[anime_id].semantic_id for anime_id in anime_ids[:3]]
            prompt = (
                "A user who liked "
                f"{sid_values[0]} and {sid_values[1]} and {sid_values[2]} "
                "would enjoy these anime because they share"
            )
            completion = _build_reasoning_text(anime_ids=anime_ids[:3], anime_meta=anime_meta)
        elif kind == "single":
            sid = semantic_map[anime_ids[0]].semantic_id
            prompt = f"{sid} is interesting to fans of"
            completion = _build_single_anchor_reasoning(anime_id=anime_ids[0], anime_meta=anime_meta)
        else:
            continue

        rows.append(
            {
                "task": "C",
                "input": prompt,
                "output": completion,
                "anime_ids": anime_ids,
                "user_id": user_id,
                "meta": {"template_id": f"reasoning_{kind}", "kind": kind},
            }
        )
        kind_counts[kind] += 1

    stats = {
        "rows_generated": len(rows),
        "kind_counts": dict(sorted(kind_counts.items())),
        "skipped_missing_ids": skipped_missing_ids,
    }
    return rows, stats


def _build_reasoning_text(*, anime_ids: Iterable[int], anime_meta: dict[int, AnimeMeta]) -> str:
    metas = [anime_meta.get(anime_id) for anime_id in anime_ids]
    present = [meta for meta in metas if meta is not None]
    if not present:
        return "overlapping tone, pacing, and character-driven conflicts"

    genre_sets = [_split_csv(meta.genres) for meta in present]
    non_empty_genre_sets = [item for item in genre_sets if item]
    shared_genres: list[str] = []
    top_genres: list[str] = []
    if non_empty_genre_sets:
        shared = set(non_empty_genre_sets[0])
        for values in non_empty_genre_sets[1:]:
            shared &= set(values)
        shared_genres = sorted(shared)
        if not shared_genres:
            freq = Counter(genre for values in non_empty_genre_sets for genre in values)
            top_genres = [name for name, _count in freq.most_common(3)]

    studio_sets = [_split_csv(meta.studios) for meta in present if meta.studios]
    shared_studios: list[str] = []
    if studio_sets:
        shared = set(studio_sets[0])
        for values in studio_sets[1:]:
            shared &= set(values)
        shared_studios = sorted(shared)

    years = [meta.year for meta in present if meta.year is not None]
    types = [meta.anime_type for meta in present if meta.anime_type]
    unique_types = sorted(set(types))

    clauses: list[str] = []
    if shared_genres:
        clauses.append(f"{_join_phrases(shared_genres[:3])} themes")
    elif top_genres:
        clauses.append(f"{_join_phrases(top_genres[:3])} elements")
    if shared_studios:
        clauses.append(f"production style from {_join_phrases(shared_studios[:2])}")
    if years:
        years_sorted = sorted(years)
        if years_sorted[0] == years_sorted[-1]:
            clauses.append(f"a similar {years_sorted[0]} era tone")
        elif years_sorted[-1] - years_sorted[0] <= 6:
            clauses.append(f"closely aligned {years_sorted[0]}-{years_sorted[-1]} era pacing")
    if len(unique_types) == 1:
        clauses.append(f"{unique_types[0].lower()} storytelling rhythms")

    if not clauses:
        return "overlapping tone, pacing, and character-driven conflicts"
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _build_single_anchor_reasoning(*, anime_id: int, anime_meta: dict[int, AnimeMeta]) -> str:
    meta = anime_meta.get(anime_id)
    if meta is None:
        return "character-focused stories, strong atmosphere, and thematic depth"
    genres = _split_csv(meta.genres)
    source = meta.source.lower() if meta.source else ""
    anime_type = meta.anime_type.lower() if meta.anime_type else ""
    clauses: list[str] = []
    if genres:
        clauses.append(_join_phrases(genres[:3]))
    if anime_type:
        clauses.append(f"{anime_type} storytelling")
    if source:
        clauses.append(f"{source}-inspired worldbuilding")
    if not clauses:
        return "character-focused stories, strong atmosphere, and thematic depth"
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _themes_from_genres(genres: str) -> str:
    parts = _split_csv(genres)
    if not parts:
        return ""
    selected = parts[:3]
    return ", ".join(selected)


def _sample_to_quota(rows: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    if not rows:
        raise RuntimeError(f"Cannot satisfy quota={quota} because source pool is empty.")

    if len(rows) >= quota:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [dict(rows[index]) for index in indices[:quota]]

    sampled: list[dict[str, Any]] = []
    while len(sampled) < quota:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for index in indices:
            sampled.append(dict(rows[index]))
            if len(sampled) >= quota:
                break
    return sampled


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    return text


def _coerce_nonneg_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    parts = [chunk.strip() for chunk in value.replace(";", ",").split(",")]
    deduped = [part for part in parts if part]
    return list(dict.fromkeys(deduped))


def _join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
