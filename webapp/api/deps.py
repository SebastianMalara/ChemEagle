from __future__ import annotations

from pathlib import Path

from review_service import get_review_service

from .runtime import DEFAULT_REVIEW_DB_PATH


def resolve_review_db_path(review_db_path: str = "") -> str:
    target = review_db_path or DEFAULT_REVIEW_DB_PATH
    return str(Path(target).expanduser().resolve())


def review_service_for(review_db_path: str = ""):
    return get_review_service(resolve_review_db_path(review_db_path))
