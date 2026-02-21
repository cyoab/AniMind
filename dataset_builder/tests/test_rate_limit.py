from __future__ import annotations

import time

import pytest

from animind_dataset.rate_limit import AsyncRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_never_exceeds_three_per_second() -> None:
    limiter = AsyncRateLimiter(max_calls=3, period_seconds=1.0)
    timestamps: list[float] = []

    for _ in range(6):
        await limiter.acquire()
        timestamps.append(time.monotonic())

    assert timestamps[3] - timestamps[0] >= 0.95
    assert timestamps[5] - timestamps[2] >= 0.95
