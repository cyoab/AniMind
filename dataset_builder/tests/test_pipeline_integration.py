from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from animind_dataset.config import Settings
from animind_dataset.pipeline import run_build
from animind_dataset.storage_sqlite import DatasetStorage
from tests.fakes import FakeClientConfig, FakeJikanClient


REQUIRED_OUTPUTS = {
    "users.parquet",
    "user_anime_list.parquet",
    "completed_ratings.parquet",
    "anime.parquet",
    "anime_statistics.parquet",
    "anime_staff.parquet",
    "anime_reviews.parquet",
    "run_manifest.json",
    "state.sqlite3",
}


def assert_outputs(out_dir: Path) -> None:
    produced = {path.name for path in out_dir.iterdir()}
    assert REQUIRED_OUTPUTS.issubset(produced)


@pytest.mark.asyncio
async def test_full_build_happy_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    result = await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(FakeClientConfig(user_count=30)),
    )

    assert result.run_id.startswith("run_")
    assert_outputs(out_dir)

    storage = DatasetStorage(out_dir / "state.sqlite3")
    stats = storage.get_stats(result.run_id)
    storage.close()

    assert stats.users_success >= 20
    assert stats.user_anime_rows > 0
    assert stats.anime_rows > 0
    assert stats.anime_reviews_rows > 0


@pytest.mark.asyncio
async def test_partial_failure_then_resume_continuation(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    run_id = "run_resume"

    await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        run_id=run_id,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(FakeClientConfig(user_count=30, fail_once_users={"user3"})),
    )

    storage = DatasetStorage(out_dir / "state.sqlite3")
    first_stats = storage.get_stats(run_id)
    storage.close()
    assert first_stats.users_errors >= 1

    await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        run_id=run_id,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(FakeClientConfig(user_count=30)),
    )

    storage = DatasetStorage(out_dir / "state.sqlite3")
    second_stats = storage.get_stats(run_id)
    storage.close()
    assert second_stats.users_errors == 0
    assert second_stats.users_success >= 20


@pytest.mark.asyncio
async def test_private_and_deleted_users_are_recorded_and_skipped(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"

    result = await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(
            FakeClientConfig(
                user_count=25,
                private_users={"user1", "user2"},
                missing_users={"user3"},
            )
        ),
    )

    storage = DatasetStorage(out_dir / "state.sqlite3")
    stats = storage.get_stats(result.run_id)
    storage.close()

    assert stats.users_private >= 2
    assert stats.users_not_found >= 1


@pytest.mark.asyncio
async def test_anime_dedup_integrity_across_reruns(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    run_id = "run_dedup"

    await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        run_id=run_id,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(FakeClientConfig(user_count=30, fail_once_users={"user4"})),
    )

    await run_build(
        out_dir=out_dir,
        target_users=20,
        include_nsfw=True,
        review_limit=50,
        run_id=run_id,
        settings=Settings(concurrency=4, rate_limit_per_second=100),
        client=FakeJikanClient(FakeClientConfig(user_count=30)),
    )

    conn = sqlite3.connect(out_dir / "state.sqlite3")
    duplicate_review_keys = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT review_key, COUNT(*) n
            FROM anime_reviews
            WHERE run_id=?
            GROUP BY review_key
            HAVING n > 1
        )
        """,
        (run_id,),
    ).fetchone()[0]
    duplicate_staff_keys = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT mal_id, person_mal_id, COUNT(*) n
            FROM anime_staff
            WHERE run_id=?
            GROUP BY mal_id, person_mal_id
            HAVING n > 1
        )
        """,
        (run_id,),
    ).fetchone()[0]
    conn.close()

    assert duplicate_review_keys == 0
    assert duplicate_staff_keys == 0
