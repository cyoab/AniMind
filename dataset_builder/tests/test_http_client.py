from __future__ import annotations

import json

import httpx
import pytest

from animind_dataset.config import Settings
from animind_dataset.http_client import JikanClient, JikanTemporaryError, RequestTelemetry


@pytest.mark.asyncio
async def test_retry_backoff_on_429_then_success() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text=json.dumps({"data": [], "pagination": {"has_next_page": False}}))

    telemetry = RequestTelemetry()
    client = JikanClient(
        settings=Settings(rate_limit_per_second=100, max_retries=2),
        telemetry=telemetry,
        transport=httpx.MockTransport(handler),
    )

    payload = await client.fetch_users_page(1)
    await client.close()

    assert payload["data"] == []
    assert call_count["n"] == 2
    assert telemetry.rate_limited == 1
    assert telemetry.retries == 1


@pytest.mark.asyncio
async def test_raises_after_exhausting_5xx_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = JikanClient(
        settings=Settings(rate_limit_per_second=100, max_retries=1),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(JikanTemporaryError):
        await client.fetch_users_page(1)

    await client.close()
