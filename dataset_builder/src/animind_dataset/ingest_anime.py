from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .http_client import JikanForbiddenError, JikanNotFoundError, JikanTemporaryError
from .models import parse_anime_core, parse_anime_reviews, parse_anime_staff, parse_anime_stats
from .storage_sqlite import DatasetStorage


@dataclass(slots=True)
class AnimeProgress:
    processed: int = 0
    success: int = 0
    skipped_nsfw: int = 0
    errors: int = 0
    reviews_ingested: int = 0


AnimeCallback = Callable[[AnimeProgress, int], None] | None


def _is_nsfw_rating(rating: str | None) -> bool:
    if rating is None:
        return False
    normalized = rating.lower()
    return normalized.startswith("rx") or "hentai" in normalized


async def ingest_anime_enrichment(
    *,
    client,
    storage: DatasetStorage,
    run_id: str,
    include_nsfw: bool,
    review_limit: int,
    concurrency: int,
    callback: AnimeCallback = None,
) -> AnimeProgress:
    anime_ids = storage.get_pending_anime_ids(run_id)
    progress = AnimeProgress()
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def update(success_type: str, reviews_added: int = 0) -> None:
        async with progress_lock:
            progress.processed += 1
            if success_type == "success":
                progress.success += 1
            elif success_type == "skipped_nsfw":
                progress.skipped_nsfw += 1
            else:
                progress.errors += 1
            progress.reviews_ingested += reviews_added
            if callback:
                callback(progress, len(anime_ids))

    async def worker(anime_mal_id: int) -> None:
        async with semaphore:
            try:
                core_data = await client.fetch_anime_full(anime_mal_id)
                core = parse_anime_core(core_data)
                nsfw = _is_nsfw_rating(core.rating)
                if nsfw and not include_nsfw:
                    storage.mark_anime_processed(run_id, anime_mal_id, "skipped_nsfw")
                    await update("skipped_nsfw")
                    return

                stats_data = await client.fetch_anime_statistics(anime_mal_id)
                staff_data = await client.fetch_anime_staff(anime_mal_id)
                review_data = await client.fetch_anime_reviews(anime_mal_id, limit=review_limit)

                stats = parse_anime_stats(stats_data, anime_mal_id)
                staff = parse_anime_staff(staff_data, anime_mal_id)
                reviews = parse_anime_reviews(review_data, anime_mal_id)

                storage.upsert_anime_core(run_id, core, nsfw=nsfw)
                storage.upsert_anime_stats(run_id, stats)
                storage.replace_anime_staff(run_id, anime_mal_id, staff)
                storage.upsert_anime_reviews(run_id, reviews)
                storage.mark_anime_processed(run_id, anime_mal_id, "success")
                await update("success", reviews_added=len(reviews))
            except JikanNotFoundError:
                storage.mark_anime_processed(run_id, anime_mal_id, "not_found", "anime not found")
                await update("error")
            except JikanForbiddenError:
                storage.mark_anime_processed(run_id, anime_mal_id, "private", "forbidden")
                await update("error")
            except JikanTemporaryError as exc:
                storage.mark_anime_processed(run_id, anime_mal_id, "rate_limited", str(exc))
                await update("error")
            except Exception as exc:  # pragma: no cover
                storage.mark_anime_processed(run_id, anime_mal_id, "error", str(exc))
                await update("error")

    await asyncio.gather(*(worker(anime_id) for anime_id in anime_ids))
    return progress
