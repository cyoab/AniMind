from __future__ import annotations

from animind_dataset.models import make_review_key, parse_user_animelist


def test_animelist_status_mapping_and_filtering() -> None:
    rows = parse_user_animelist(
        "alice",
        [
            {
                "anime": {"mal_id": 1},
                "status": "Completed",
                "score": 8,
                "episodes_watched": 12,
            },
            {
                "anime": {"mal_id": 2},
                "status": "on hold",
                "score": 0,
                "episodes_watched": 4,
            },
            {
                "anime": {"mal_id": 3},
                "status": "unknown_status",
                "score": 7,
                "episodes_watched": 10,
            },
        ],
    )

    assert len(rows) == 2
    assert rows[0].status == "completed"
    assert rows[1].status == "on_hold"


def test_review_key_fallback_is_stable() -> None:
    key1 = make_review_key(1, None, "bob", "hello")
    key2 = make_review_key(1, None, "bob", "hello")
    key3 = make_review_key(1, 99, "bob", "hello")

    assert key1 == key2
    assert key3 == "1:99"
