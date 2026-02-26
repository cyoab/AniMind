from __future__ import annotations

import csv
import json
import random
import sqlite3
from array import array
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
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
from rich.table import Table

from .embed import _cuda_unavailable_reason, _resolve_device
from .rqvae import load_rqvae_for_eval


@dataclass(slots=True)
class TokenizeConfig:
    tokenizer_db: Path = Path("./output/tokenizer.sqlite")
    source_db: Path = Path("../output/anilist.sqlite")
    rqvae_checkpoint: Path = Path("./output/rqvae/rqvae_best.pt")
    out_dir: Path = Path("./output/tokenize")
    rebuild: bool = True
    limit: int = 0
    device: str = "auto"
    batch_size: int = 512
    write_db_tables: bool = False
    special_tokens: list[str] = field(
        default_factory=lambda: [
            "<anime_start>",
            "<anime_end>",
            "<watch>",
            "<liked>",
            "<disliked>",
            "<dropped>",
        ]
    )
    semantic_id_concat: bool = True
    semantic_id_separator: str = ""
    cluster_sample_size: int = 10
    cluster_min_bucket: int = 20
    cluster_random_seed: int = 42
    recall_k: int = 20
    recall_min_support: int = 10
    recall_positive_score_min: int = 7
    recall_completed_status: int = 2
    recall_max_queries: int = 500
    recall_max_rows: int = 0
    recall_seed: int = 42
    dry_run: bool = False
    dry_run_limit: int = 512
    dry_run_out_subdir: str = "dry_run"

@dataclass(slots=True)
class _AnimeEmbeddingRow:
    anime_id: int
    name: str
    english_name: str
    genres: str


def run_tokenize(config: TokenizeConfig) -> None:
    console = Console()
    effective_config = _apply_dry_run_overrides(config=config, console=console)

    if not effective_config.tokenizer_db.exists():
        raise RuntimeError(f"Tokenizer DB not found: {effective_config.tokenizer_db}")
    if not effective_config.rqvae_checkpoint.exists():
        raise RuntimeError(f"RQ-VAE checkpoint not found: {effective_config.rqvae_checkpoint}")

    requested_device = effective_config.device.strip().lower()
    resolved_device = _resolve_device(device=effective_config.device, torch_module=torch)
    if requested_device == "auto" and resolved_device == "cpu":
        console.log(
            "[yellow]CUDA unavailable; falling back to CPU.[/yellow] "
            f"{_cuda_unavailable_reason(torch_module=torch)}"
        )

    effective_config.out_dir.mkdir(parents=True, exist_ok=True)
    _reset_output_artifacts(config=effective_config)
    tokenized_at = _now()

    stage_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    console.rule("[bold cyan]Semantic ID Tokenization[/bold cyan]")
    with Progress(*stage_columns, console=console) as stage_progress:
        stage_task = stage_progress.add_task("[cyan]Pipeline stages[/cyan]", total=7)

        stage_progress.update(stage_task, description="[cyan]1/7 Load embeddings[/cyan]")
        rows, vectors = _load_embeddings_with_metadata(
            tokenizer_db=effective_config.tokenizer_db,
            limit=effective_config.limit,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]2/7 Load model/runtime[/cyan]")
        model, checkpoint = load_rqvae_for_eval(
            checkpoint_path=effective_config.rqvae_checkpoint,
            device=resolved_device,
        )
        model.eval()
        rq_levels, codebook_size = _extract_checkpoint_quantizer_shape(checkpoint=checkpoint)
        gpu_name, gpu_vram = _verify_runtime(resolved_device=resolved_device)
        runtime_table = _build_runtime_table(
            config=effective_config,
            resolved_device=resolved_device,
            row_count=len(rows),
            input_dim=int(vectors.size(1)),
            rq_levels=rq_levels,
            codebook_size=codebook_size,
            gpu_name=gpu_name,
            gpu_vram=gpu_vram,
        )
        console.print(runtime_table)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]3/7 Quantize embeddings[/cyan]")
        codes = _quantize_codes(
            model=model,
            vectors=vectors,
            batch_size=effective_config.batch_size,
            device=resolved_device,
            console=console,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(
            stage_task,
            description="[cyan]4/7 Build vocab and Semantic IDs[/cyan]",
        )
        semantic_rows, observed_counts = _build_semantic_rows(
            rows=rows,
            codes=codes,
            separator=effective_config.semantic_id_separator,
            concat=effective_config.semantic_id_concat,
        )
        vocab = _build_vocab(
            rq_levels=rq_levels,
            codebook_size=codebook_size,
            special_tokens=effective_config.special_tokens,
            observed_counts=observed_counts,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]5/7 Write artifacts[/cyan]")
        _write_artifacts(
            config=effective_config,
            tokenized_at=tokenized_at,
            semantic_rows=semantic_rows,
            vocab=vocab,
            rq_levels=rq_levels,
            codebook_size=codebook_size,
            checkpoint=checkpoint,
        )
        if effective_config.write_db_tables:
            _write_db_tables(
                tokenizer_db=effective_config.tokenizer_db,
                semantic_rows=semantic_rows,
                vocab_tokens=vocab["tokens"],
                checkpoint_path=effective_config.rqvae_checkpoint,
                tokenized_at=tokenized_at,
            )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]6/7 Cluster sanity checks[/cyan]")
        cluster_summary = _run_cluster_inspection(
            out_dir=effective_config.out_dir,
            semantic_rows=semantic_rows,
            sample_size=effective_config.cluster_sample_size,
            min_bucket=effective_config.cluster_min_bucket,
            seed=effective_config.cluster_random_seed,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]7/7 Recall evaluation[/cyan]")
        recall_report = _run_recall_eval(
            config=effective_config,
            semantic_rows=semantic_rows,
            codes=codes,
        )
        summary_payload = {
            "tokenized_at": tokenized_at,
            "row_count": len(semantic_rows),
            "rq_levels": rq_levels,
            "codebook_size": codebook_size,
            "checkpoint_path": str(effective_config.rqvae_checkpoint),
            "cluster_summary": cluster_summary,
            "recall_summary": recall_report,
        }
        _write_json(effective_config.out_dir / "tokenize_run_summary.json", summary_payload)
        if effective_config.dry_run:
            _validate_dry_run_artifacts(out_dir=effective_config.out_dir)
        stage_progress.advance(stage_task)

    console.log(
        "Tokenize complete: "
        f"rows={len(semantic_rows):,}, "
        f"semantic_ids={effective_config.out_dir / 'semantic_ids.jsonl'}, "
        f"vocab_size={len(vocab['tokens']):,}."
    )


def _apply_dry_run_overrides(*, config: TokenizeConfig, console: Console) -> TokenizeConfig:
    if not config.dry_run:
        return config
    limited_rows = max(2, int(config.dry_run_limit))
    if config.limit > 0:
        limited_rows = min(limited_rows, int(config.limit))
    dry_run_config = replace(
        config,
        limit=limited_rows,
        recall_max_rows=(
            min(config.recall_max_rows, 100_000) if config.recall_max_rows > 0 else 100_000
        ),
        out_dir=config.out_dir / config.dry_run_out_subdir,
    )
    console.log(
        "[yellow]Dry-run mode enabled.[/yellow] "
        f"rows<={dry_run_config.limit}, out_dir={dry_run_config.out_dir}."
    )
    return dry_run_config


def _reset_output_artifacts(config: TokenizeConfig) -> None:
    if not config.rebuild:
        return
    artifact_paths = (
        config.out_dir / "semantic_ids.jsonl",
        config.out_dir / "semantic_lookup.tsv",
        config.out_dir / "semantic_id_to_anime.jsonl",
        config.out_dir / "semantic_vocab.json",
        config.out_dir / "special_tokens.json",
        config.out_dir / "token_stats.json",
        config.out_dir / "cluster_inspection.jsonl",
        config.out_dir / "cluster_summary.json",
        config.out_dir / "recall_report.json",
        config.out_dir / "tokenize_config.json",
        config.out_dir / "tokenize_run_summary.json",
    )
    for path in artifact_paths:
        if path.exists():
            path.unlink()


def _load_embeddings_with_metadata(
    *,
    tokenizer_db: Path,
    limit: int,
) -> tuple[list[_AnimeEmbeddingRow], torch.Tensor]:
    query = """
        SELECT
            e.anime_id,
            e.embedding,
            e.embedding_dim,
            p.name,
            COALESCE(p.english_name, ''),
            COALESCE(p.genres, '')
        FROM anime_embeddings e
        INNER JOIN anime_prep p ON p.anime_id = e.anime_id
        ORDER BY e.anime_id
    """
    if limit > 0:
        query += f" LIMIT {int(limit)}"

    rows: list[_AnimeEmbeddingRow] = []
    vectors: list[torch.Tensor] = []
    expected_dim: int | None = None
    with sqlite3.connect(tokenizer_db) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        if not _table_exists(conn, "anime_embeddings"):
            raise RuntimeError(
                "Missing anime_embeddings table. Run --phase embed before --phase tokenize."
            )
        if not _table_exists(conn, "anime_prep"):
            raise RuntimeError(
                "Missing anime_prep table. Run --phase prep before --phase tokenize."
            )

        cursor = conn.execute(query)
        for anime_id, embedding_blob, embedding_dim, name, english_name, genres in cursor:
            dim = int(embedding_dim)
            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise RuntimeError("anime_embeddings contains inconsistent embedding_dim values.")
            vector = _blob_to_tensor(blob=embedding_blob)
            if int(vector.numel()) != dim:
                raise RuntimeError("Embedding blob length does not match embedding_dim metadata.")
            if not bool(torch.isfinite(vector).all()):
                raise RuntimeError(
                    "anime_embeddings contains non-finite values (NaN/Inf). "
                    f"anime_id={int(anime_id)}"
                )
            vectors.append(vector)
            rows.append(
                _AnimeEmbeddingRow(
                    anime_id=int(anime_id),
                    name=str(name),
                    english_name=str(english_name),
                    genres=str(genres),
                )
            )
    if not rows:
        raise RuntimeError("No rows found in joined anime_embeddings/anime_prep dataset.")
    return rows, torch.stack(vectors, dim=0)


def _quantize_codes(
    *,
    model: Any,
    vectors: torch.Tensor,
    batch_size: int,
    device: str,
    console: Console,
) -> torch.Tensor:
    code_batches: list[torch.Tensor] = []
    total_rows = int(vectors.size(0))
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with torch.no_grad(), Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Quantizing embeddings[/cyan]", total=total_rows)
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = vectors[start:end].to(device, non_blocking=True)
            latent = model.encoder(batch)
            _, _, codes, _ = model.quantizer(latent)
            code_batches.append(codes.detach().cpu().to(torch.int16))
            progress.advance(task, advance=(end - start))
    return torch.cat(code_batches, dim=0).to(torch.int32)


def _build_semantic_rows(
    *,
    rows: list[_AnimeEmbeddingRow],
    codes: torch.Tensor,
    separator: str,
    concat: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    semantic_rows: list[dict[str, Any]] = []
    observed_counts: dict[str, int] = {}
    for idx, row in enumerate(rows):
        code_row = [int(v) for v in codes[idx].tolist()]
        tokens = [f"<L{level_idx + 1}_{code}>" for level_idx, code in enumerate(code_row)]
        semantic_id = "".join(tokens) if concat else separator.join(tokens)
        for token in tokens:
            observed_counts[token] = observed_counts.get(token, 0) + 1
        semantic_rows.append(
            {
                "anime_id": row.anime_id,
                "name": row.name,
                "english_name": row.english_name,
                "genres": row.genres,
                "semantic_id": semantic_id,
                "tokens": tokens,
                "codes": code_row,
            }
        )
    return semantic_rows, observed_counts


def _build_vocab(
    *,
    rq_levels: int,
    codebook_size: int,
    special_tokens: list[str],
    observed_counts: dict[str, int],
) -> dict[str, Any]:
    level_tokens: list[str] = []
    for level in range(1, rq_levels + 1):
        for code in range(codebook_size):
            level_tokens.append(f"<L{level}_{code}>")
    deduped_special = list(
        dict.fromkeys([token.strip() for token in special_tokens if token.strip()])
    )
    vocab_tokens = deduped_special + level_tokens
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    observed_level_tokens = sorted(observed_counts.items(), key=lambda item: item[0])
    return {
        "tokens": vocab_tokens,
        "special_tokens": deduped_special,
        "token_to_id": token_to_id,
        "observed_counts": observed_level_tokens,
        "rq_levels": rq_levels,
        "codebook_size": codebook_size,
    }


def _write_artifacts(
    *,
    config: TokenizeConfig,
    tokenized_at: str,
    semantic_rows: list[dict[str, Any]],
    vocab: dict[str, Any],
    rq_levels: int,
    codebook_size: int,
    checkpoint: dict[str, Any],
) -> None:
    _write_jsonl(config.out_dir / "semantic_ids.jsonl", semantic_rows)

    with (config.out_dir / "semantic_lookup.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "anime_id",
                "name",
                "english_name",
                "semantic_id",
                "level_tokens_json",
                "codes_json",
            ]
        )
        for row in semantic_rows:
            writer.writerow(
                [
                    row["anime_id"],
                    row["name"],
                    row["english_name"],
                    row["semantic_id"],
                    json.dumps(row["tokens"], ensure_ascii=True),
                    json.dumps(row["codes"], ensure_ascii=True),
                ]
            )

    inverse_rows = [
        {
            "semantic_id": row["semantic_id"],
            "anime_id": row["anime_id"],
            "name": row["name"],
            "english_name": row["english_name"],
        }
        for row in semantic_rows
    ]
    _write_jsonl(config.out_dir / "semantic_id_to_anime.jsonl", inverse_rows)

    _write_json(
        config.out_dir / "semantic_vocab.json",
        {
            "tokenized_at": tokenized_at,
            "rq_levels": rq_levels,
            "codebook_size": codebook_size,
            "vocab_size": len(vocab["tokens"]),
            "special_tokens": vocab["special_tokens"],
            "tokens": vocab["tokens"],
            "token_to_id": vocab["token_to_id"],
        },
    )
    _write_json(config.out_dir / "special_tokens.json", {"special_tokens": vocab["special_tokens"]})
    _write_json(
        config.out_dir / "token_stats.json",
        {
            "observed_counts": vocab["observed_counts"],
            "observed_token_total": int(sum(int(item[1]) for item in vocab["observed_counts"])),
        },
    )
    _write_json(
        config.out_dir / "tokenize_config.json",
        {
            **asdict(config),
            "tokenizer_db": str(config.tokenizer_db),
            "source_db": str(config.source_db),
            "rqvae_checkpoint": str(config.rqvae_checkpoint),
            "out_dir": str(config.out_dir),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        },
    )


def _write_db_tables(
    *,
    tokenizer_db: Path,
    semantic_rows: list[dict[str, Any]],
    vocab_tokens: list[str],
    checkpoint_path: Path,
    tokenized_at: str,
) -> None:
    with sqlite3.connect(tokenizer_db) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        conn.execute("DROP TABLE IF EXISTS anime_semantic_ids")
        conn.execute("DROP TABLE IF EXISTS semantic_vocab")
        conn.execute(
            """
            CREATE TABLE anime_semantic_ids (
                anime_id INTEGER PRIMARY KEY,
                semantic_id TEXT NOT NULL,
                level_tokens_json TEXT NOT NULL,
                codes_json TEXT NOT NULL,
                source_checkpoint TEXT NOT NULL,
                tokenized_at TEXT NOT NULL,
                name TEXT,
                english_name TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE semantic_vocab (
                token_id INTEGER PRIMARY KEY,
                token TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO anime_semantic_ids(
                anime_id, semantic_id, level_tokens_json, codes_json,
                source_checkpoint, tokenized_at, name, english_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(row["anime_id"]),
                    str(row["semantic_id"]),
                    json.dumps(row["tokens"], ensure_ascii=True),
                    json.dumps(row["codes"], ensure_ascii=True),
                    str(checkpoint_path),
                    tokenized_at,
                    str(row["name"]),
                    str(row["english_name"]),
                )
                for row in semantic_rows
            ],
        )
        conn.executemany(
            "INSERT INTO semantic_vocab(token_id, token) VALUES (?, ?)",
            [(idx, token) for idx, token in enumerate(vocab_tokens)],
        )
        conn.commit()


def _run_cluster_inspection(
    *,
    out_dir: Path,
    semantic_rows: list[dict[str, Any]],
    sample_size: int,
    min_bucket: int,
    seed: int,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in semantic_rows:
        first_token = str(row["tokens"][0])
        buckets.setdefault(first_token, []).append(row)

    ranked = sorted(
        [(token, members) for token, members in buckets.items() if len(members) >= min_bucket],
        key=lambda item: len(item[1]),
        reverse=True,
    )
    rng = random.Random(seed)
    selected = ranked[:20]

    inspection_rows: list[dict[str, Any]] = []
    purity_values: list[float] = []
    for token, members in selected:
        pick = list(members)
        rng.shuffle(pick)
        sample = pick[:sample_size]
        avg_genre_jaccard = _average_genre_jaccard(sample=sample)
        if avg_genre_jaccard is not None:
            purity_values.append(avg_genre_jaccard)
        inspection_rows.append(
            {
                "first_token": token,
                "bucket_size": len(members),
                "avg_genre_jaccard": avg_genre_jaccard,
                "samples": [
                    {
                        "anime_id": int(row["anime_id"]),
                        "name": row["name"],
                        "english_name": row["english_name"],
                        "genres": row["genres"],
                        "semantic_id": row["semantic_id"],
                    }
                    for row in sample
                ],
            }
        )
    _write_jsonl(out_dir / "cluster_inspection.jsonl", inspection_rows)
    summary = {
        "bucket_min_size": min_bucket,
        "eligible_bucket_count": len(ranked),
        "inspected_bucket_count": len(inspection_rows),
        "global_avg_genre_jaccard": (mean(purity_values) if purity_values else None),
    }
    _write_json(out_dir / "cluster_summary.json", summary)
    return summary


def _average_genre_jaccard(*, sample: list[dict[str, Any]]) -> float | None:
    sets = [_parse_genres(str(row["genres"])) for row in sample]
    filtered = [item for item in sets if item]
    if len(filtered) < 2:
        return None
    scores: list[float] = []
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            a = filtered[i]
            b = filtered[j]
            denom = len(a | b)
            if denom == 0:
                continue
            scores.append(float(len(a & b)) / float(denom))
    if not scores:
        return None
    return mean(scores)


def _parse_genres(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _run_recall_eval(
    *,
    config: TokenizeConfig,
    semantic_rows: list[dict[str, Any]],
    codes: torch.Tensor,
) -> dict[str, Any]:
    report_path = config.out_dir / "recall_report.json"
    if not config.source_db.exists():
        report = {
            "status": "skipped",
            "reason": f"source_db not found: {config.source_db}",
        }
        _write_json(report_path, report)
        return report

    target_ids = {int(row["anime_id"]) for row in semantic_rows}
    user_to_items, item_to_users = _build_positive_watch_graph(
        source_db=config.source_db,
        target_ids=target_ids,
        completed_status=config.recall_completed_status,
        score_min=config.recall_positive_score_min,
        max_rows=config.recall_max_rows,
    )
    support = {anime_id: len(users) for anime_id, users in item_to_users.items()}
    eligible = [anime_id for anime_id, cnt in support.items() if cnt >= config.recall_min_support]
    if not eligible:
        report = {
            "status": "skipped",
            "reason": "No eligible anime for recall evaluation after support filtering.",
            "min_support": config.recall_min_support,
            "total_positive_users": len(user_to_items),
            "target_anime_count": len(target_ids),
        }
        _write_json(report_path, report)
        return report

    rng = random.Random(config.recall_seed)
    query_ids = list(eligible)
    rng.shuffle(query_ids)
    if config.recall_max_queries > 0:
        query_ids = query_ids[: config.recall_max_queries]

    anime_ids = [int(row["anime_id"]) for row in semantic_rows]
    id_to_index = {anime_id: idx for idx, anime_id in enumerate(anime_ids)}
    codes_cpu = codes.to(torch.int32).cpu()
    recall_values: list[float] = []
    examples: list[dict[str, Any]] = []
    for anime_id in query_ids:
        query_index = id_to_index.get(anime_id)
        if query_index is None:
            continue
        proxy = _proxy_neighbors_for_anime(
            anime_id=anime_id,
            k=config.recall_k,
            user_to_items=user_to_items,
            item_to_users=item_to_users,
            support=support,
        )
        if not proxy:
            continue
        semantic = _semantic_neighbors_for_anime(
            query_index=query_index,
            anime_ids=anime_ids,
            codes=codes_cpu,
            k=config.recall_k,
        )
        proxy_ids = [item[0] for item in proxy]
        semantic_ids = [item[0] for item in semantic]
        denom = max(1, min(config.recall_k, len(proxy_ids)))
        overlap = len(set(proxy_ids) & set(semantic_ids))
        value = float(overlap) / float(denom)
        recall_values.append(value)
        examples.append(
            {
                "anime_id": anime_id,
                "recall_at_k": value,
                "proxy_neighbors": proxy_ids[:5],
                "semantic_neighbors": semantic_ids[:5],
            }
        )

    if not recall_values:
        report = {
            "status": "skipped",
            "reason": "No queries produced both proxy and semantic neighbors.",
            "eligible_queries": len(query_ids),
        }
        _write_json(report_path, report)
        return report

    sorted_examples = sorted(examples, key=lambda item: float(item["recall_at_k"]))
    report = {
        "status": "ok",
        "recall_k": config.recall_k,
        "completed_status": config.recall_completed_status,
        "score_min": config.recall_positive_score_min,
        "min_support": config.recall_min_support,
        "eligible_queries": len(query_ids),
        "evaluated_queries": len(recall_values),
        "mean_recall_at_k": mean(recall_values),
        "median_recall_at_k": median(recall_values),
        "best_examples": sorted_examples[-5:],
        "worst_examples": sorted_examples[:5],
    }
    _write_json(report_path, report)
    return report


def _build_positive_watch_graph(
    *,
    source_db: Path,
    target_ids: set[int],
    completed_status: int,
    score_min: int,
    max_rows: int,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    user_to_items: dict[int, set[int]] = {}
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        if not _table_exists(conn, "user_anime"):
            raise RuntimeError(f"Missing user_anime table in source_db: {source_db}")
        query = """
            SELECT user_id, anime_id
            FROM user_anime
            WHERE watching_status = ? AND score >= ?
        """
        if max_rows > 0:
            query += f" LIMIT {int(max_rows)}"
        cursor = conn.execute(query, (completed_status, score_min))
        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            for user_id, anime_id in batch:
                uid = int(user_id)
                aid = int(anime_id)
                if aid not in target_ids:
                    continue
                user_to_items.setdefault(uid, set()).add(aid)

    item_to_users: dict[int, set[int]] = {}
    for user_id, items in user_to_items.items():
        for anime_id in items:
            item_to_users.setdefault(anime_id, set()).add(user_id)
    return user_to_items, item_to_users


def _proxy_neighbors_for_anime(
    *,
    anime_id: int,
    k: int,
    user_to_items: dict[int, set[int]],
    item_to_users: dict[int, set[int]],
    support: dict[int, int],
) -> list[tuple[int, float]]:
    users = item_to_users.get(anime_id, set())
    if not users:
        return []
    co_counts: dict[int, int] = {}
    for user_id in users:
        for candidate in user_to_items.get(user_id, set()):
            if candidate == anime_id:
                continue
            co_counts[candidate] = co_counts.get(candidate, 0) + 1

    support_i = support.get(anime_id, 0)
    if support_i <= 0:
        return []
    ranked: list[tuple[int, float]] = []
    for candidate, co in co_counts.items():
        support_j = support.get(candidate, 0)
        denom = support_i + support_j - co
        if denom <= 0:
            continue
        score = float(co) / float(denom)
        ranked.append((candidate, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def _semantic_neighbors_for_anime(
    *,
    query_index: int,
    anime_ids: list[int],
    codes: torch.Tensor,
    k: int,
) -> list[tuple[int, tuple[float, int]]]:
    qcodes = codes[query_index]
    matches = torch.eq(codes, qcodes.unsqueeze(0))
    mismatch_count = torch.logical_not(matches).sum(dim=1).to(torch.float32)
    levels = max(1, int(codes.size(1)))
    distances = mismatch_count / float(levels)

    prefix_flags = torch.cumprod(matches.to(torch.int32), dim=1)
    prefix_depth = prefix_flags.sum(dim=1).to(torch.int32)

    ranking: list[tuple[int, float, int]] = []
    for idx, anime_id in enumerate(anime_ids):
        if idx == query_index:
            continue
        ranking.append(
            (
                int(anime_id),
                float(distances[idx].item()),
                int(prefix_depth[idx].item()),
            )
        )
    ranking.sort(key=lambda item: (item[1], -item[2], item[0]))
    return [(anime_id, (distance, prefix)) for anime_id, distance, prefix in ranking[:k]]


def _extract_checkpoint_quantizer_shape(*, checkpoint: dict[str, Any]) -> tuple[int, int]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint missing config metadata.")
    rq_levels = int(config.get("rq_levels", 0))
    codebook_size = int(config.get("codebook_size", 0))
    if rq_levels < 1 or codebook_size < 1:
        raise RuntimeError("Checkpoint config has invalid rq_levels/codebook_size.")
    return rq_levels, codebook_size


def _build_runtime_table(
    *,
    config: TokenizeConfig,
    resolved_device: str,
    row_count: int,
    input_dim: int,
    rq_levels: int,
    codebook_size: int,
    gpu_name: str | None,
    gpu_vram: float | None,
) -> Table:
    table = Table(title="Tokenize Runtime Summary", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Checkpoint", str(config.rqvae_checkpoint))
    table.add_row("Resolved Device", resolved_device)
    table.add_row("Rows", f"{row_count:,}")
    table.add_row("Input Dim", str(input_dim))
    table.add_row("Levels / Codebook", f"{rq_levels} / {codebook_size}")
    table.add_row("Batch Size", str(config.batch_size))
    if gpu_name is not None and gpu_vram is not None:
        table.add_row("GPU", f"{gpu_name} ({gpu_vram:.2f} GiB)")
    return table


def _verify_runtime(resolved_device: str) -> tuple[str | None, float | None]:
    if not resolved_device.startswith("cuda"):
        return None, None
    target_device = torch.device(resolved_device)
    cuda_index = (
        int(target_device.index)
        if target_device.index is not None
        else int(torch.cuda.current_device())
    )
    props = torch.cuda.get_device_properties(cuda_index)
    probe = torch.randn((128, 128), device=resolved_device, dtype=torch.float32)
    _ = (probe @ probe.T).sum().item()
    torch.cuda.synchronize()
    return props.name, props.total_memory / (1024**3)


def _validate_dry_run_artifacts(*, out_dir: Path) -> None:
    required = (
        out_dir / "semantic_ids.jsonl",
        out_dir / "semantic_lookup.tsv",
        out_dir / "semantic_vocab.json",
        out_dir / "recall_report.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Dry-run validation failed; missing artifacts: {', '.join(missing)}")


def _blob_to_tensor(*, blob: bytes) -> torch.Tensor:
    unpacked = array("f")
    unpacked.frombytes(blob)
    return torch.tensor(unpacked, dtype=torch.float32)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
