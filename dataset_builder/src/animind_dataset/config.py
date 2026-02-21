from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class Settings:
    base_url: str = os.getenv("ANIMIND_API_BASE_URL", "https://api.jikan.moe/v4")
    rate_limit_per_second: int = int(os.getenv("ANIMIND_RATE_LIMIT_PER_SECOND", "3"))
    rate_limit_period_seconds: float = 1.0
    max_retries: int = int(os.getenv("ANIMIND_MAX_RETRIES", "5"))
    timeout_seconds: float = float(os.getenv("ANIMIND_TIMEOUT_SECONDS", "30"))
    concurrency: int = int(os.getenv("ANIMIND_CONCURRENCY", "6"))
    user_buffer_ratio: float = float(os.getenv("ANIMIND_USER_BUFFER_RATIO", "0.15"))
    default_review_limit: int = int(os.getenv("ANIMIND_REVIEW_LIMIT", "50"))
