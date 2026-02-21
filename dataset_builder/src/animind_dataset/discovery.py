from __future__ import annotations

from typing import Callable

from .storage_sqlite import DatasetStorage


DiscoveryCallback = Callable[[int, int, int, str], None] | None


async def discover_users(
    *,
    client,
    storage: DatasetStorage,
    run_id: str,
    target_users: int,
    buffer_ratio: float,
    callback: DiscoveryCallback = None,
) -> int:
    goal = max(target_users, int(target_users * (1.0 + buffer_ratio)))
    current = storage.count_users(run_id)
    if current >= goal:
        return current

    page = int(storage.get_checkpoint(run_id, "last_user_page", "1") or "1")
    stale_pages = 0

    while current < goal and stale_pages < 25:
        payload = await client.fetch_users_page(page)
        users_payload = payload.get("data") or []
        inserted = storage.upsert_discovered_users(run_id, users_payload)
        current = storage.count_users(run_id)
        storage.set_checkpoint(run_id, "last_user_page", str(page))
        if callback:
            callback(current, goal, page, "users")

        stale_pages = stale_pages + 1 if inserted == 0 else 0

        pagination = payload.get("pagination") or {}
        if not pagination.get("has_next_page"):
            break
        page += 1

    random_attempts = 0
    max_random_attempts = max(300, target_users * 3)
    while current < goal and random_attempts < max_random_attempts:
        random_user = await client.fetch_random_user()
        storage.upsert_discovered_users(run_id, [random_user])
        current = storage.count_users(run_id)
        random_attempts += 1
        if callback:
            callback(current, goal, random_attempts, "random")

    storage.set_checkpoint(run_id, "discovery_done", "1")
    return current
