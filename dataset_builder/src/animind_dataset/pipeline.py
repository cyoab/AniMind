from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import uuid

from .config import Settings
from .discovery import discover_users
from .export_parquet import export_run_to_parquet
from .http_client import JikanClient, RequestTelemetry
from .ingest_anime import AnimeProgress, ingest_anime_enrichment
from .ingest_users import ingest_user_preferences
from .models import UserProgress
from .storage_sqlite import COMPLETED, FAILED, DatasetStorage


@dataclass(slots=True)
class BuildResult:
    run_id: str
    db_path: Path
    out_dir: Path
    manifest_path: Path
    telemetry: RequestTelemetry
    user_progress: UserProgress
    anime_progress: AnimeProgress


def _make_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run_{stamp}_{suffix}"


async def run_build(
    *,
    out_dir: Path,
    target_users: int,
    include_nsfw: bool,
    review_limit: int,
    run_id: str | None = None,
    settings: Settings | None = None,
    discovery_callback=None,
    user_callback=None,
    anime_callback=None,
    client=None,
) -> BuildResult:
    settings = settings or Settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "state.sqlite3"

    storage = DatasetStorage(db_path)
    storage.initialize()

    resolved_run_id = run_id or _make_run_id()
    if not storage.run_exists(resolved_run_id):
        storage.create_run(resolved_run_id, target_users, include_nsfw, review_limit)

    telemetry = RequestTelemetry()

    owned_client = client is None
    try:
        async with AsyncExitStack() as stack:
            if client is None:
                client = JikanClient(settings=settings, telemetry=telemetry)
                await stack.enter_async_context(client)

            try:
                await discover_users(
                    client=client,
                    storage=storage,
                    run_id=resolved_run_id,
                    target_users=target_users,
                    buffer_ratio=settings.user_buffer_ratio,
                    callback=discovery_callback,
                )
                user_progress = await ingest_user_preferences(
                    client=client,
                    storage=storage,
                    run_id=resolved_run_id,
                    target_users=target_users,
                    concurrency=settings.concurrency,
                    callback=user_callback,
                )
                user_anime_rows = storage.count_user_anime_rows(resolved_run_id)
                if user_anime_rows == 0:
                    raise RuntimeError(
                        "No user anime list rows were ingested. "
                        "The run is incomplete (likely endpoint mismatch, private users, or sustained rate limiting)."
                    )
                anime_progress = await ingest_anime_enrichment(
                    client=client,
                    storage=storage,
                    run_id=resolved_run_id,
                    include_nsfw=include_nsfw,
                    review_limit=review_limit,
                    concurrency=settings.concurrency,
                    callback=anime_callback,
                )
                storage.set_run_status(resolved_run_id, COMPLETED)
            except Exception:
                storage.set_run_status(resolved_run_id, FAILED)
                raise
            finally:
                if not owned_client and hasattr(client, "close"):
                    close_result = client.close()
                    if asyncio.iscoroutine(close_result):
                        await close_result

        export_run_to_parquet(db_path=db_path, run_id=resolved_run_id, out_dir=out_dir)
    finally:
        storage.close()

    return BuildResult(
        run_id=resolved_run_id,
        db_path=db_path,
        out_dir=out_dir,
        manifest_path=out_dir / "run_manifest.json",
        telemetry=telemetry,
        user_progress=user_progress,
        anime_progress=anime_progress,
    )
