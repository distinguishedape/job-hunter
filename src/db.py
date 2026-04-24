"""SQLite persistence using sqlite-utils.  Handles dedup by job ID (hash of url+title)."""

from __future__ import annotations

from pathlib import Path

import sqlite_utils
import structlog

from src.models import JobPosting, JobScore

log = structlog.get_logger(__name__)


def get_db(path: str = "jobs.db") -> sqlite_utils.Database:
    db = sqlite_utils.Database(path)
    _ensure_schema(db)
    return db


def _ensure_schema(db: sqlite_utils.Database) -> None:
    if "jobs" not in db.table_names():
        db["jobs"].create(
            {
                "id": str,
                "source": str,
                "company": str,
                "title": str,
                "location": str,
                "url": str,
                "description": str,
                "posted_date": str,
                "scraped_at": str,
            },
            pk="id",
        )
        db["jobs"].create_index(["source"], if_not_exists=True)
        log.info("created_table", table="jobs")

    if "scores" not in db.table_names():
        db["scores"].create(
            {
                "job_id": str,
                "fit_score": int,
                "reasons": str,       # JSON array stored as text
                "strengths": str,
                "red_flags": str,
                "should_apply": int,  # SQLite has no bool
            },
            pk="job_id",
        )
        log.info("created_table", table="scores")


def upsert_job(db: sqlite_utils.Database, job: JobPosting) -> bool:
    """Insert job; return True if it was new, False if already present."""
    if db["jobs"].get(job.id) if _row_exists(db, "jobs", job.id) else None:  # type: ignore[index]
        return False

    row = job.model_dump()
    row["id"] = job.id
    row["scraped_at"] = job.scraped_at.isoformat()

    db["jobs"].upsert(row, pk="id")  # type: ignore[index]
    log.info("job_saved", id=job.id, company=job.company, title=job.title)
    return True


def is_new_job(db: sqlite_utils.Database, job: JobPosting) -> bool:
    return not _row_exists(db, "jobs", job.id)


def upsert_score(db: sqlite_utils.Database, score: JobScore) -> None:
    import json

    row = {
        "job_id": score.job_id,
        "fit_score": score.fit_score,
        "reasons": json.dumps(score.reasons),
        "strengths": json.dumps(score.strengths),
        "red_flags": json.dumps(score.red_flags),
        "should_apply": int(score.should_apply),
    }
    db["scores"].upsert(row, pk="job_id")  # type: ignore[index]
    log.info("score_saved", job_id=score.job_id, fit_score=score.fit_score)


def get_unscored_jobs(db: sqlite_utils.Database) -> list[dict]:
    return list(
        db.execute(
            "SELECT j.* FROM jobs j LEFT JOIN scores s ON j.id = s.job_id WHERE s.job_id IS NULL"
        ).fetchall()
    )


def _row_exists(db: sqlite_utils.Database, table: str, pk: str) -> bool:
    try:
        db[table].get(pk)  # type: ignore[index]
        return True
    except sqlite_utils.db.NotFoundError:
        return False
