"""Pure-function tests for the Telegram notifier — no network."""

from __future__ import annotations

from src.models import JobPosting, JobScore
from src.notifier import _esc, format_message


def test_esc_escapes_all_reserved_chars() -> None:
    # Every MarkdownV2 reserved char must be backslash-prefixed.
    raw = "_*[]()~`>#+-=|{}.!\\"
    escaped = _esc(raw)
    for ch in raw:
        assert f"\\{ch}" in escaped, f"missing escape for {ch!r} in {escaped!r}"


def test_esc_leaves_plain_text_alone() -> None:
    assert _esc("hello world") == "hello world"


def test_format_message_contains_key_fields() -> None:
    job = JobPosting(
        source="yc",
        company="Acme (YC W24)",
        title="Founding ML Engineer",
        location="Bengaluru",
        url="https://example.com/jobs/1",
        description="desc",
    )
    score = JobScore(
        job_id=job.id,
        fit_score=8,
        reasons=["a", "b"],
        strengths=["Python", "LLMs"],
        red_flags=["senior title"],
        should_apply=True,
    )
    msg = format_message(job, score)

    assert "Score 8/10" in msg
    # Escaped company parens + dashes
    assert "Acme \\(YC W24\\)" in msg
    assert "Founding ML Engineer" in msg
    assert "Bengaluru" in msg
    # URL dot is escaped
    assert "example\\.com" in msg
    assert "Python; LLMs" in msg
    assert "senior title" in msg


def test_format_message_no_red_flags_shows_none() -> None:
    job = JobPosting(
        source="hn", company="X", title="MLE", location="Remote",
        url="https://x.test/j", description="d",
    )
    score = JobScore(
        job_id=job.id, fit_score=9, reasons=["r1", "r2"],
        strengths=["match"], red_flags=[], should_apply=True,
    )
    msg = format_message(job, score)
    assert "none" in msg
