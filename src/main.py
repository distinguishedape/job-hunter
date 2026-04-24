"""End-to-end orchestrator.

Runs weekly (via GitHub Actions) or on demand (`uv run python -m src.main`).

Flow:
    1. Run all four scrapers concurrently; one source failing doesn't kill the others.
    2. Upsert each posting into SQLite — the hash-based PK dedupes across sources
       and across weeks, so we only score each job once.
    3. Score *new* jobs with the LLM (Groq Llama 3.3). Scores are persisted so a
       re-run never re-scores, which keeps the free-tier bill at $0.
    4. For anything scoring >= threshold, send a Telegram message.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

import litellm
import structlog

from src.config import Settings, get_settings
from src.db import get_db, get_unscored_jobs, upsert_job, upsert_score
from src.models import JobPosting, JobScore
from src.notifier import notify_matches
from src.scorer import _load_cv, _make_client, score_job
from src.scrapers.base import Scraper
from src.scrapers.hasjob import HasjobScraper
from src.scrapers.hn import HNScraper
from src.scrapers.internshala import InternshalaScraper
from src.scrapers.yc import YCScraper

log = structlog.get_logger(__name__)

# Groq free tier caps at 12k tokens/minute for llama-3.3-70b; each scoring call
# is ~2k tokens, so a ~12s baseline gap keeps us comfortably under the limit.
_PACE_SECONDS = 12.0
# When Groq does rate-limit us, its error string contains "try again in <N>s".
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


async def _score_with_ratelimit_retry(
    job: JobPosting,
    *,
    settings: Settings,
    cv_text: str,
    client: Any,
) -> tuple[JobScore, float]:
    """Score one job, retrying once if Groq rate-limits us.

    Groq's 429 message includes a precise "try again in Xs" hint — we parse it
    and sleep exactly that long (plus a small buffer) rather than guessing.
    """
    try:
        return score_job(job, settings=settings, cv_text=cv_text, client=client)
    except litellm.RateLimitError as e:
        wait_s = _parse_retry_after(str(e))
        log.warning("rate_limited", job_id=job.id, sleeping=wait_s)
        await asyncio.sleep(wait_s + 1.0)
        return score_job(job, settings=settings, cv_text=cv_text, client=client)


def _parse_retry_after(error_message: str) -> float:
    m = _RETRY_AFTER_RE.search(error_message)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 30.0  # Groq's minute-window fallback


def _build_scrapers() -> list[Scraper]:
    return [YCScraper(), HasjobScraper(), InternshalaScraper(), HNScraper()]


async def _run_scrapers(scrapers: list[Scraper]) -> list[JobPosting]:
    """Run every scraper concurrently; swallow per-source exceptions."""
    results = await asyncio.gather(
        *(s.scrape() for s in scrapers),
        return_exceptions=True,
    )

    jobs: list[JobPosting] = []
    for scraper, result in zip(scrapers, results, strict=True):
        if isinstance(result, BaseException):
            log.warning("scraper_failed", source=scraper.name, error=str(result))
            continue
        log.info("scraper_ok", source=scraper.name, count=len(result))
        jobs.extend(result)
    return jobs


async def run(settings: Settings | None = None) -> dict:
    """Scrape → dedupe → score new jobs → notify matches. Returns run summary."""
    settings = settings or get_settings()
    db = get_db(settings.db_path)

    log.info("run_start")

    # 1. Scrape concurrently
    all_jobs = await _run_scrapers(_build_scrapers())
    log.info("scraped_total", count=len(all_jobs))

    # 2. Upsert every scraped job. The hash-based PK dedupes silently.
    for job in all_jobs:
        upsert_job(db, job)

    # 3. Score every job in the db that doesn't have a score yet. Sourcing the
    #    work list from the db (rather than "new this run") lets us pick up any
    #    jobs that were scraped but unscored in a previous run — e.g. after a
    #    rate-limit crash — without re-scraping or re-scoring anything.
    unscored_rows = get_unscored_jobs(db)
    jobs_to_score = [_row_to_job(r) for r in unscored_rows]
    log.info("dedupe_done", scraped=len(all_jobs), to_score=len(jobs_to_score))

    if not jobs_to_score:
        log.info("run_done", matches=0, reason="nothing_to_score")
        return {"scraped": len(all_jobs), "scored": 0, "matches": 0, "notified": 0, "cost_usd": 0.0}

    # Persist each score the moment we get it so a crash (or a rate-limit wedge)
    # doesn't throw away the work we've already paid for.
    cv_text = _load_cv(settings.cv_path)
    client = _make_client(settings.groq_api_key.get_secret_value())

    total_cost = 0.0
    scored_count = 0
    matches: list[tuple[JobPosting, JobScore]] = []

    for i, job in enumerate(jobs_to_score):
        try:
            score, cost = await _score_with_ratelimit_retry(
                job, settings=settings, cv_text=cv_text, client=client
            )
        except Exception as e:
            log.warning("score_skipped", job_id=job.id, company=job.company, error=str(e))
            continue

        upsert_score(db, score)
        total_cost += cost
        scored_count += 1
        if score.fit_score >= settings.score_threshold:
            matches.append((job, score))

        # Baseline pace between calls so we don't trip Groq's TPM limit.
        if i < len(jobs_to_score) - 1:
            await asyncio.sleep(_PACE_SECONDS)

    log.info(
        "scoring_done",
        scored=scored_count,
        matches=len(matches),
        threshold=settings.score_threshold,
        total_cost_usd=round(total_cost, 6),
    )

    # 4. Notify
    notified = await notify_matches(matches, settings=settings)

    summary = {
        "scraped": len(all_jobs),
        "scored": scored_count,
        "matches": len(matches),
        "notified": notified,
        "cost_usd": round(total_cost, 6),
    }
    log.info("run_done", **summary)
    return summary


def _row_to_job(row: dict | tuple) -> JobPosting:
    """Rebuild a JobPosting from a sqlite row (sqlite-utils returns tuples)."""
    # Column order from SELECT j.* on the jobs table created in db.py
    cols = ["id", "source", "company", "title", "location", "url",
            "description", "posted_date", "scraped_at"]
    data = dict(zip(cols, row, strict=True)) if isinstance(row, tuple) else dict(row)
    return JobPosting(
        source=data["source"],
        company=data["company"],
        title=data["title"],
        location=data["location"],
        url=data["url"],
        description=data["description"] or "",
        posted_date=data.get("posted_date"),
    )


def main() -> int:
    try:
        summary = asyncio.run(run())
    except Exception as e:
        log.exception("run_failed", error=str(e))
        return 1

    print(
        f"\nRun summary: scraped={summary['scraped']} "
        f"scored={summary['scored']} matches={summary['matches']} "
        f"notified={summary['notified']} cost=${summary['cost_usd']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
