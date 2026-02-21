from __future__ import annotations

import asyncio
from dataclasses import dataclass
import random
import time
from typing import Any

import httpx

from .config import Settings
from .rate_limit import AsyncRateLimiter


class JikanError(Exception):
    pass


class JikanNotFoundError(JikanError):
    pass


class JikanForbiddenError(JikanError):
    pass


class JikanTemporaryError(JikanError):
    pass


@dataclass(slots=True)
class RequestTelemetry:
    total_requests: int = 0
    successful_requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    errors: int = 0
    total_latency_seconds: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency_seconds / self.total_requests) * 1000.0


class JikanClient:
    def __init__(
        self,
        settings: Settings,
        telemetry: RequestTelemetry | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry or RequestTelemetry()
        self._limiter = AsyncRateLimiter(
            max_calls=settings.rate_limit_per_second,
            period_seconds=settings.rate_limit_period_seconds,
        )
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            headers={"User-Agent": "animind-dataset/0.1"},
            transport=transport,
        )

    async def __aenter__(self) -> JikanClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        backoff_seconds = 0.5

        for attempt in range(self.settings.max_retries + 1):
            await self._limiter.acquire()
            started_at = time.monotonic()
            self.telemetry.total_requests += 1
            try:
                response = await self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                self.telemetry.errors += 1
                if attempt >= self.settings.max_retries:
                    raise JikanTemporaryError(str(exc)) from exc
                self.telemetry.retries += 1
                await asyncio.sleep(backoff_seconds * (2**attempt) + random.random() * 0.1)
                continue
            finally:
                self.telemetry.total_latency_seconds += time.monotonic() - started_at

            if response.status_code == 200:
                self.telemetry.successful_requests += 1
                return response.json()

            if response.status_code == 404:
                raise JikanNotFoundError(path)
            if response.status_code in (401, 403):
                raise JikanForbiddenError(path)

            if response.status_code == 429:
                self.telemetry.rate_limited += 1
                if attempt >= self.settings.max_retries:
                    raise JikanTemporaryError(f"429 Too Many Requests for {path}")
                retry_after = response.headers.get("Retry-After")
                retry_after_seconds = float(retry_after) if retry_after else backoff_seconds * (2**attempt)
                self.telemetry.retries += 1
                await asyncio.sleep(retry_after_seconds + random.random() * 0.1)
                continue

            if 500 <= response.status_code < 600:
                self.telemetry.errors += 1
                if attempt >= self.settings.max_retries:
                    raise JikanTemporaryError(f"{response.status_code} for {path}")
                self.telemetry.retries += 1
                await asyncio.sleep(backoff_seconds * (2**attempt) + random.random() * 0.1)
                continue

            response.raise_for_status()

        raise JikanTemporaryError(f"Exhausted retries for {path}")

    async def fetch_users_page(self, page: int) -> dict[str, Any]:
        return await self._get_json("/users", params={"page": page})

    async def fetch_random_user(self) -> dict[str, Any]:
        payload = await self._get_json("/random/users")
        return payload.get("data") or {}

    async def fetch_user_full(self, username: str) -> dict[str, Any]:
        payload = await self._get_json(f"/users/{username}/full")
        return payload.get("data") or {}

    async def fetch_user_animelist(self, username: str) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while True:
            payload = await self._get_json(
                f"/users/{username}/animelist",
                params={"status": "all", "page": page},
            )
            rows.extend(payload.get("data") or [])
            pagination = payload.get("pagination") or {}
            if not pagination.get("has_next_page"):
                break
            page += 1
        return rows

    async def fetch_user_reviews(self, username: str, max_pages: int = 1) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while page <= max_pages:
            payload = await self._get_json(
                f"/users/{username}/reviews",
                params={"type": "anime", "page": page},
            )
            rows.extend(payload.get("data") or [])
            pagination = payload.get("pagination") or {}
            if not pagination.get("has_next_page"):
                break
            page += 1
        return rows

    async def fetch_anime_full(self, anime_mal_id: int) -> dict[str, Any]:
        payload = await self._get_json(f"/anime/{anime_mal_id}/full")
        return payload.get("data") or {}

    async def fetch_anime_statistics(self, anime_mal_id: int) -> dict[str, Any]:
        payload = await self._get_json(f"/anime/{anime_mal_id}/statistics")
        return payload.get("data") or {}

    async def fetch_anime_staff(self, anime_mal_id: int) -> list[dict[str, Any]]:
        payload = await self._get_json(f"/anime/{anime_mal_id}/staff")
        return payload.get("data") or []

    async def fetch_anime_reviews(self, anime_mal_id: int, limit: int = 50) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while len(rows) < limit:
            payload = await self._get_json(f"/anime/{anime_mal_id}/reviews", params={"page": page})
            data = payload.get("data") or []
            rows.extend(data)
            if len(rows) >= limit:
                return rows[:limit]
            pagination = payload.get("pagination") or {}
            if not pagination.get("has_next_page"):
                break
            page += 1
        return rows[:limit]
