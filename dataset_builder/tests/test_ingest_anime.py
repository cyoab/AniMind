from __future__ import annotations

import json

import httpx
import pytest

from animind_dataset.config import Settings
from animind_dataset.http_client import JikanClient


@pytest.mark.asyncio
async def test_anime_review_pagination_stops_at_limit_50() -> None:
    seen_pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/anime/1/reviews":
            page = int(request.url.params.get("page", "1"))
            seen_pages.append(page)
            has_next = page == 1
            data = [
                {
                    "mal_id": page * 100 + i,
                    "user": {"username": f"u{i}"},
                    "review": f"r{i}",
                }
                for i in range(30)
            ]
            return httpx.Response(
                200,
                text=json.dumps({"data": data, "pagination": {"has_next_page": has_next}}),
            )
        return httpx.Response(404)

    client = JikanClient(
        settings=Settings(rate_limit_per_second=100),
        transport=httpx.MockTransport(handler),
    )

    reviews = await client.fetch_anime_reviews(1, limit=50)
    await client.close()

    assert len(reviews) == 50
    assert seen_pages == [1, 2]
