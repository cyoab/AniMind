from __future__ import annotations

import asyncio
from collections.abc import Callable

from .http_client import JikanForbiddenError, JikanNotFoundError, JikanTemporaryError
from .models import UserProgress, parse_user_animelist, parse_user_profile
from .storage_sqlite import DatasetStorage


UserCallback = Callable[[UserProgress, int], None] | None


async def ingest_user_preferences(
    *,
    client,
    storage: DatasetStorage,
    run_id: str,
    target_users: int,
    concurrency: int,
    callback: UserCallback = None,
) -> UserProgress:
    usernames = storage.get_usernames_for_ingestion(run_id, limit=target_users)
    progress = UserProgress()
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def update_progress(on_success: str | None = None) -> None:
        async with progress_lock:
            progress.processed += 1
            if on_success == "success":
                progress.success += 1
            elif on_success == "private":
                progress.private += 1
            elif on_success == "not_found":
                progress.not_found += 1
            elif on_success == "error":
                progress.errors += 1
            if callback:
                callback(progress, len(usernames))

    async def worker(username: str) -> None:
        async with semaphore:
            try:
                profile_data = await client.fetch_user_full(username)
                profile = parse_user_profile(profile_data, fallback_username=username)
                storage.upsert_user_profile(run_id, profile)
            except JikanNotFoundError:
                storage.set_user_crawl_state(run_id, username, "not_found", "user full not found")
                await update_progress("not_found")
                return
            except JikanForbiddenError:
                storage.set_user_crawl_state(run_id, username, "private", "profile forbidden")
                await update_progress("private")
                return
            except JikanTemporaryError as exc:
                storage.set_user_crawl_state(run_id, username, "rate_limited", str(exc))
                await update_progress("error")
                return
            except Exception as exc:  # pragma: no cover
                storage.set_user_crawl_state(run_id, username, "error", str(exc))
                await update_progress("error")
                return

            try:
                animelist_items = await client.fetch_user_animelist(username)
                entries = parse_user_animelist(username, animelist_items)
                storage.upsert_user_anime_entries(run_id, entries)
                storage.enqueue_anime_ids(run_id, [entry.anime_mal_id for entry in entries])
                # Optional signal for user-linked review preference context.
                user_reviews = await client.fetch_user_reviews(username, max_pages=1)
                storage.upsert_user_reviews(run_id, username, user_reviews)
                storage.set_user_crawl_state(run_id, username, "success")
                await update_progress("success")
            except JikanNotFoundError:
                storage.set_user_crawl_state(run_id, username, "not_found", "animelist not found")
                await update_progress("not_found")
            except JikanForbiddenError:
                storage.set_user_crawl_state(run_id, username, "private", "animelist forbidden")
                await update_progress("private")
            except JikanTemporaryError as exc:
                storage.set_user_crawl_state(run_id, username, "rate_limited", str(exc))
                await update_progress("error")
            except Exception as exc:  # pragma: no cover
                storage.set_user_crawl_state(run_id, username, "error", str(exc))
                await update_progress("error")

    await asyncio.gather(*(worker(username) for username in usernames))
    return progress
