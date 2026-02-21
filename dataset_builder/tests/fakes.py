from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from animind_dataset.http_client import JikanForbiddenError, JikanNotFoundError, JikanTemporaryError


@dataclass(slots=True)
class FakeClientConfig:
    user_count: int = 30
    private_users: set[str] | None = None
    missing_users: set[str] | None = None
    fail_once_users: set[str] | None = None


class FakeJikanClient:
    def __init__(self, config: FakeClientConfig | None = None) -> None:
        self.config = config or FakeClientConfig()
        self.private_users = self.config.private_users or set()
        self.missing_users = self.config.missing_users or set()
        self.fail_once_users = self.config.fail_once_users or set()
        self._fail_once_counter = defaultdict(int)

    async def close(self) -> None:
        return None

    async def fetch_users_page(self, page: int):
        if page > 1:
            return {"data": [], "pagination": {"has_next_page": False}}
        users = [
            {
                "username": f"user{i}",
                "mal_id": i,
                "url": f"https://myanimelist.net/profile/user{i}",
                "images": {},
            }
            for i in range(1, self.config.user_count + 1)
        ]
        return {
            "data": users,
            "pagination": {"has_next_page": False},
        }

    async def fetch_random_user(self):
        return {
            "username": "random_user",
            "mal_id": 999999,
            "url": "https://myanimelist.net/profile/random_user",
            "images": {},
        }

    async def fetch_user_full(self, username: str):
        if username in self.missing_users:
            raise JikanNotFoundError(username)
        if username in self.private_users:
            raise JikanForbiddenError(username)
        return {
            "username": username,
            "mal_id": int(username.replace("user", "") or 0),
            "url": f"https://myanimelist.net/profile/{username}",
            "images": {},
            "joined": "2020-01-01T00:00:00+00:00",
            "last_online": "2025-01-01T00:00:00+00:00",
            "private": False,
        }

    async def fetch_user_animelist(self, username: str):
        if username in self.fail_once_users:
            self._fail_once_counter[username] += 1
            if self._fail_once_counter[username] == 1:
                raise JikanTemporaryError(f"temporary failure for {username}")

        seed = int(username.replace("user", "") or 1)
        anime_a = (seed % 7) + 1
        anime_b = ((seed + 3) % 7) + 1
        return [
            {
                "anime": {"mal_id": anime_a},
                "status": "completed",
                "score": 8,
                "episodes_watched": 12,
                "updated_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "anime": {"mal_id": anime_b},
                "status": "watching",
                "score": 0,
                "episodes_watched": 4,
                "updated_at": "2024-01-02T00:00:00+00:00",
            },
        ]

    async def fetch_user_reviews(self, username: str, max_pages: int = 1):
        return [
            {
                "mal_id": 1000 + int(username.replace("user", "") or 0),
                "entry": {"mal_id": 1},
                "score": 8,
                "review": f"Review by {username}",
                "date": "2024-01-01T00:00:00+00:00",
            }
        ]

    async def fetch_anime_full(self, anime_mal_id: int):
        return {
            "mal_id": anime_mal_id,
            "title": f"Anime {anime_mal_id}",
            "titles": [{"title": f"Anime {anime_mal_id}"}],
            "type": "TV",
            "source": "Manga",
            "episodes": 12,
            "status": "Finished Airing",
            "duration": "24 min per ep",
            "rating": "PG-13 - Teens 13 or older",
            "synopsis": f"Synopsis {anime_mal_id}",
            "genres": [{"name": "Action"}],
            "themes": [{"name": "School"}],
            "demographics": [{"name": "Shounen"}],
            "studios": [{"name": "Studio A"}],
            "producers": [{"name": "Producer A"}],
        }

    async def fetch_anime_statistics(self, anime_mal_id: int):
        return {
            "watching": 10,
            "completed": 100,
            "on_hold": 3,
            "dropped": 4,
            "plan_to_watch": 25,
            "total": 142,
            "scores": [{"score": 8, "votes": 50}, {"score": 9, "votes": 30}],
        }

    async def fetch_anime_staff(self, anime_mal_id: int):
        return [
            {
                "person": {"mal_id": 200 + anime_mal_id, "name": f"Staff {anime_mal_id}"},
                "positions": ["Director"],
            }
        ]

    async def fetch_anime_reviews(self, anime_mal_id: int, limit: int = 50):
        rows = []
        for i in range(min(limit, 3)):
            rows.append(
                {
                    "mal_id": anime_mal_id * 100 + i,
                    "user": {"username": f"reviewer{i}"},
                    "score": 8,
                    "tags": ["story"],
                    "is_spoiler": False,
                    "is_preliminary": False,
                    "review": f"Review {i} for anime {anime_mal_id}",
                    "reactions": {"overall": 1},
                    "date": "2024-01-01T00:00:00+00:00",
                }
            )
        return rows
