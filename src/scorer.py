"""LLM-based job scorer using Instructor + LiteLLM (Groq by default).

Each call logs the estimated cost via structlog so you can track spend per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import instructor
import litellm
import structlog

from src.config import Settings, get_settings
from src.models import JobPosting, JobScore

log = structlog.get_logger(__name__)

_SYSTEM_TEMPLATE = """\
You are an expert career advisor helping a job-seeker evaluate whether a role is a good fit.

## Candidate CV
{cv_text}

## Scoring Rubric (weight each dimension equally)
1. **Stack overlap** — does the job's tech stack match the candidate's skills?
2. **Seniority match** — is the role appropriate for an intern / entry-level candidate?
   Penalise roles requiring 3+ years of experience.
3. **Company stage** — early-stage AI-native startups score higher than large corporates.
4. **Location** — role must be in Bengaluru / Bangalore / remote-friendly India.
   Penalise roles that are US/EU on-site only.

Return a JSON object matching the JobScore schema exactly.
fit_score must be an integer 1-10.
should_apply should be True only when fit_score >= 7 and there are no hard disqualifiers.
"""

_USER_TEMPLATE = """\
## Job to evaluate
Company: {company}
Title: {title}
Location: {location}
Source: {source}
URL: {url}

### Description
{description}

Score this job against the candidate's CV using the rubric above.
"""


def _load_cv(cv_path: str) -> str:
    path = Path(cv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CV not found at '{cv_path}'. Copy cv.example.md → cv.md and fill it in."
        )
    return path.read_text(encoding="utf-8")


def _make_client(api_key: str) -> Any:
    """Return an Instructor-wrapped LiteLLM client."""
    litellm.api_key = api_key  # LiteLLM picks up GROQ_API_KEY from env too
    return instructor.from_litellm(litellm.completion)


def score_job(
    job: JobPosting,
    *,
    settings: Settings | None = None,
    cv_text: str | None = None,
    client: Any | None = None,
) -> tuple[JobScore, float]:
    """Score a single job posting.

    Returns (JobScore, estimated_cost_usd).
    Pass `client` and `cv_text` in tests to avoid real LLM calls.
    """
    # Load settings lazily — skip entirely when client + cv_text are injected (e.g. tests)
    if settings is None and (cv_text is None or client is None):
        settings = get_settings()

    if cv_text is None:
        cv_text = _load_cv(settings.cv_path)  # type: ignore[union-attr]

    if client is None:
        client = _make_client(settings.groq_api_key.get_secret_value())  # type: ignore[union-attr]

    model = settings.llm_model if settings else "groq/llama-3.3-70b-versatile"

    system_prompt = _SYSTEM_TEMPLATE.format(cv_text=cv_text)
    user_prompt = _USER_TEMPLATE.format(
        company=job.company,
        title=job.title,
        location=job.location,
        source=job.source,
        url=job.url,
        description=job.description[:4000],  # stay well within context limits
    )

    score, raw_response = client.chat.completions.create_with_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=JobScore,
        max_retries=2,
    )

    # Stamp the job_id (Instructor can't know it before parsing)
    score.job_id = job.id

    cost = _extract_cost(raw_response, model)
    log.info(
        "job_scored",
        job_id=job.id,
        company=job.company,
        title=job.title,
        fit_score=score.fit_score,
        should_apply=score.should_apply,
        cost_usd=round(cost, 6),
    )

    return score, cost


def _extract_cost(raw_response: Any, model: str) -> float:
    """Best-effort cost extraction from a LiteLLM response object."""
    try:
        return litellm.completion_cost(completion_response=raw_response)
    except Exception:
        # Groq free tier reports $0; if cost extraction fails, return 0
        return 0.0


def score_jobs_batch(
    jobs: list[JobPosting],
    *,
    settings: Settings | None = None,
    cv_text: str | None = None,
    client: Any | None = None,
) -> list[tuple[JobPosting, JobScore, float]]:
    """Score a list of jobs sequentially. Returns (job, score, cost) triples."""
    if settings is None and (cv_text is None or client is None):
        settings = get_settings()
    if cv_text is None:
        cv_text = _load_cv(settings.cv_path)  # type: ignore[union-attr]
    if client is None:
        client = _make_client(settings.groq_api_key.get_secret_value())  # type: ignore[union-attr]

    results = []
    total_cost = 0.0
    for job in jobs:
        score, cost = score_job(job, settings=settings, cv_text=cv_text, client=client)
        total_cost += cost
        results.append((job, score, cost))

    log.info("batch_scored", count=len(jobs), total_cost_usd=round(total_cost, 6))
    return results
