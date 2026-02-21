from __future__ import annotations

import asyncio
import time
from collections import deque


class AsyncRateLimiter:
    """Simple sliding-window limiter to keep requests within provider constraints."""

    def __init__(self, max_calls: int, period_seconds: float = 1.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._calls = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period_seconds:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait_seconds = self.period_seconds - (now - self._calls[0])
            await asyncio.sleep(max(wait_seconds, 0.0))
