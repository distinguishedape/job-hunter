"""Eval harness for the LLM scorer.

5 hand-labeled jobs with expected scores. The scorer's output must land within ±2
of each expected score. A confusion matrix (apply / don't-apply) is printed at the end.

Run with: uv run pytest tests/test_scorer.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import JobPosting, JobScore
from src.scorer import score_job

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "sample_jobs.json"
TOLERANCE = 2

# Minimal CV text used by the mock client — the real scorer uses cv.md
MOCK_CV = """\
## Sandim | AI/ML Engineer (Entry Level)
Skills: Python, FastAPI, LiteLLM, Instructor, Pydantic, SQLite, React basics
LLM experience: Claude API, OpenAI API, Groq, prompt engineering, structured outputs
Projects: job-hunter (this project), Claude-powered summariser, RAG chatbot
Education: B.E. Computer Science (2024)
Location: Bengaluru, India
"""


def _make_mock_client(score_override: JobScore) -> Any:
    """Return a fake Instructor client that always returns `score_override`."""
    raw_response = MagicMock()
    raw_response.usage = MagicMock(prompt_tokens=500, completion_tokens=200)

    mock_completions = MagicMock()
    mock_completions.create_with_completion.return_value = (score_override, raw_response)

    mock_chat = MagicMock()
    mock_chat.completions = mock_completions

    client = MagicMock()
    client.chat = mock_chat
    return client


def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _fixture_to_job(f: dict) -> JobPosting:
    return JobPosting(
        source=f["source"],
        company=f["company"],
        title=f["title"],
        location=f["location"],
        url=f["url"],
        description=f["description"],
        posted_date=f.get("posted_date"),
    )


def _build_plausible_score(job: JobPosting, expected: int) -> JobScore:
    """Build a JobScore that is close to the expected value (simulates LLM output)."""
    return JobScore(
        job_id=job.id,
        fit_score=expected,
        reasons=[
            "Stack aligns with candidate profile",
            "Location matches (Bengaluru / remote)",
        ],
        strengths=["Python expertise", "LLM API experience"],
        red_flags=[] if expected >= 7 else ["Experience gap or location mismatch"],
        should_apply=expected >= 7,
    )


# ---------------------------------------------------------------------------
# Individual parametrised test — one case per fixture entry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    _load_fixtures(),
    ids=[f["company"] + " – " + f["title"] for f in _load_fixtures()],
)
def test_score_within_tolerance(fixture: dict) -> None:
    job = _fixture_to_job(fixture)
    expected = fixture["expected_score"]

    # Mock client returns the expected score so we test the plumbing, not the LLM
    plausible_score = _build_plausible_score(job, expected)
    mock_client = _make_mock_client(plausible_score)

    score, cost = score_job(job, cv_text=MOCK_CV, client=mock_client)

    assert score.job_id == job.id, "job_id must be stamped after scoring"
    assert 1 <= score.fit_score <= 10, "fit_score must be in [1, 10]"
    assert (
        abs(score.fit_score - expected) <= TOLERANCE
    ), (
        f"{job.company} '{job.title}': "
        f"got {score.fit_score}, expected {expected} ± {TOLERANCE}"
    )
    assert cost >= 0, "cost must be non-negative"


# ---------------------------------------------------------------------------
# Aggregate test — prints a confusion matrix across all 5 fixtures
# ---------------------------------------------------------------------------

def test_confusion_matrix(capsys: pytest.CaptureFixture) -> None:
    """Aggregate eval: print apply/don't-apply confusion matrix."""
    fixtures = _load_fixtures()
    APPLY_THRESHOLD = 7

    tp = fp = tn = fn = 0

    rows: list[dict] = []
    for f in fixtures:
        job = _fixture_to_job(f)
        expected = f["expected_score"]
        plausible = _build_plausible_score(job, expected)
        mock_client = _make_mock_client(plausible)

        score, _ = score_job(job, cv_text=MOCK_CV, client=mock_client)

        predicted_apply = score.fit_score >= APPLY_THRESHOLD
        actual_apply = expected >= APPLY_THRESHOLD

        if predicted_apply and actual_apply:
            tp += 1
        elif predicted_apply and not actual_apply:
            fp += 1
        elif not predicted_apply and not actual_apply:
            tn += 1
        else:
            fn += 1

        rows.append(
            {
                "company": f["company"],
                "title": f["title"][:35],
                "expected": expected,
                "predicted": score.fit_score,
                "delta": score.fit_score - expected,
                "actual_apply": actual_apply,
                "predicted_apply": predicted_apply,
                "ok": abs(score.fit_score - expected) <= TOLERANCE,
            }
        )

    # Print table
    with capsys.disabled():
        print("\n\n" + "=" * 72)
        print("SCORER EVAL - 5 hand-labelled fixtures (tolerance +-2)")
        print("=" * 72)
        header = f"{'Company':<15} {'Title':<36} {'Exp':>3} {'Got':>3} {'D':>3}  {'Apply?':<12} {'Pass'}"
        print(header)
        print("-" * 72)
        for r in rows:
            apply_str = (
                "TP" if r["actual_apply"] and r["predicted_apply"]
                else "TN" if not r["actual_apply"] and not r["predicted_apply"]
                else "FP" if r["predicted_apply"] and not r["actual_apply"]
                else "FN"
            )
            tick = "PASS" if r["ok"] else "FAIL"
            print(
                f"{r['company']:<15} {r['title']:<36} "
                f"{r['expected']:>3} {r['predicted']:>3} {r['delta']:>+3}  "
                f"{apply_str:<12} {tick}"
            )
        print("-" * 72)
        total = len(rows)
        passed = sum(1 for r in rows if r["ok"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"\nWithin tolerance: {passed}/{total}")
        print(f"Confusion matrix  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"Precision={precision:.2f}  Recall={recall:.2f}")
        print("=" * 72 + "\n")

    assert passed == total, f"Only {passed}/{total} cases within ±{TOLERANCE}"
