from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pydantic import BaseModel, Field


VALID_USER_STATUSES = {
    "watching": "watching",
    "completed": "completed",
    "on_hold": "on_hold",
    "on hold": "on_hold",
    "onhold": "on_hold",
    "dropped": "dropped",
    "plan_to_watch": "plan_to_watch",
    "plan to watch": "plan_to_watch",
    "plantowatch": "plan_to_watch",
}


class UserProfile(BaseModel):
    username: str
    mal_id: int | None = None
    profile_url: str | None = None
    images: dict[str, Any] = Field(default_factory=dict)
    joined: str | None = None
    last_online: str | None = None
    is_public: bool = True


class UserAnimeEntry(BaseModel):
    username: str
    anime_mal_id: int
    status: str
    score: int = 0
    episodes_watched: int = 0
    updated_at: str | None = None


class AnimeCore(BaseModel):
    mal_id: int
    title: str | None = None
    titles: list[str] = Field(default_factory=list)
    type: str | None = None
    source: str | None = None
    episodes: int | None = None
    status: str | None = None
    duration: str | None = None
    rating: str | None = None
    synopsis: str | None = None
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    demographics: list[str] = Field(default_factory=list)
    studios: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)


class AnimeStats(BaseModel):
    mal_id: int
    watching: int = 0
    completed: int = 0
    on_hold: int = 0
    dropped: int = 0
    plan_to_watch: int = 0
    total: int = 0
    scores_distribution: dict[str, int] = Field(default_factory=dict)


class AnimeStaffEntry(BaseModel):
    mal_id: int
    person_mal_id: int
    name: str
    positions: list[str] = Field(default_factory=list)


class AnimeReviewEntry(BaseModel):
    review_key: str
    review_mal_id: int | None = None
    anime_mal_id: int
    username: str | None = None
    score: int | None = None
    tags: list[str] = Field(default_factory=list)
    is_spoiler: bool = False
    is_preliminary: bool = False
    review: str | None = None
    reactions: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class UserProgress:
    processed: int = 0
    success: int = 0
    private: int = 0
    not_found: int = 0
    errors: int = 0


def normalize_status(raw_status: str | None) -> str | None:
    if not raw_status:
        return None
    normalized = raw_status.strip().lower()
    return VALID_USER_STATUSES.get(normalized)


def make_review_key(
    anime_mal_id: int,
    review_mal_id: int | None,
    username: str | None,
    review_text: str | None,
) -> str:
    if review_mal_id is not None:
        return f"{anime_mal_id}:{review_mal_id}"
    source = f"{anime_mal_id}|{username or ''}|{review_text or ''}"
    digest = hashlib.sha1(source.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{anime_mal_id}:h:{digest}"


def parse_user_profile(data: dict[str, Any], fallback_username: str) -> UserProfile:
    return UserProfile(
        username=data.get("username") or fallback_username,
        mal_id=data.get("mal_id"),
        profile_url=data.get("url"),
        images=data.get("images") or {},
        joined=data.get("joined"),
        last_online=data.get("last_online"),
        is_public=not bool(data.get("private", False)),
    )


def parse_user_animelist(username: str, items: list[dict[str, Any]]) -> list[UserAnimeEntry]:
    rows: list[UserAnimeEntry] = []
    for item in items:
        anime = item.get("anime") or {}
        anime_mal_id = anime.get("mal_id")
        if anime_mal_id is None:
            continue
        status = normalize_status(item.get("status"))
        if status is None:
            continue
        rows.append(
            UserAnimeEntry(
                username=username,
                anime_mal_id=int(anime_mal_id),
                status=status,
                score=int(item.get("score") or 0),
                episodes_watched=int(item.get("episodes_watched") or 0),
                updated_at=item.get("updated_at"),
            )
        )
    return rows


def _names_from_objects(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("name")) for item in items if item.get("name")]


def parse_anime_core(data: dict[str, Any]) -> AnimeCore:
    return AnimeCore(
        mal_id=int(data["mal_id"]),
        title=data.get("title"),
        titles=[str(t.get("title")) for t in data.get("titles", []) if t.get("title")],
        type=data.get("type"),
        source=data.get("source"),
        episodes=data.get("episodes"),
        status=data.get("status"),
        duration=data.get("duration"),
        rating=data.get("rating"),
        synopsis=data.get("synopsis"),
        genres=_names_from_objects(data.get("genres") or []),
        themes=_names_from_objects(data.get("themes") or []),
        demographics=_names_from_objects(data.get("demographics") or []),
        studios=_names_from_objects(data.get("studios") or []),
        producers=_names_from_objects(data.get("producers") or []),
    )


def parse_anime_stats(data: dict[str, Any], anime_mal_id: int) -> AnimeStats:
    scores_distribution = {
        str(score.get("score")): int(score.get("votes") or 0)
        for score in data.get("scores") or []
        if score.get("score") is not None
    }
    return AnimeStats(
        mal_id=anime_mal_id,
        watching=int(data.get("watching") or 0),
        completed=int(data.get("completed") or 0),
        on_hold=int(data.get("on_hold") or 0),
        dropped=int(data.get("dropped") or 0),
        plan_to_watch=int(data.get("plan_to_watch") or 0),
        total=int(data.get("total") or 0),
        scores_distribution=scores_distribution,
    )


def parse_anime_staff(items: list[dict[str, Any]], anime_mal_id: int) -> list[AnimeStaffEntry]:
    rows: list[AnimeStaffEntry] = []
    for item in items:
        person = item.get("person") or {}
        person_mal_id = person.get("mal_id")
        if person_mal_id is None:
            continue
        rows.append(
            AnimeStaffEntry(
                mal_id=anime_mal_id,
                person_mal_id=int(person_mal_id),
                name=person.get("name") or "",
                positions=[str(position) for position in item.get("positions") or []],
            )
        )
    return rows


def parse_anime_reviews(items: list[dict[str, Any]], anime_mal_id: int) -> list[AnimeReviewEntry]:
    rows: list[AnimeReviewEntry] = []
    for item in items:
        user = item.get("user") or {}
        username = user.get("username")
        review_text = item.get("review")
        review_mal_id = item.get("mal_id")
        rows.append(
            AnimeReviewEntry(
                review_key=make_review_key(anime_mal_id, review_mal_id, username, review_text),
                review_mal_id=review_mal_id,
                anime_mal_id=anime_mal_id,
                username=username,
                score=item.get("score"),
                tags=[str(tag) for tag in item.get("tags") or []],
                is_spoiler=bool(item.get("is_spoiler") or False),
                is_preliminary=bool(item.get("is_preliminary") or False),
                review=review_text,
                reactions=item.get("reactions") or {},
                created_at=item.get("date") or item.get("created_at"),
            )
        )
    return rows
