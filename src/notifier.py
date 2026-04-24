"""Telegram notifier.

For each (job, score) pair above the configured threshold, send a formatted
MarkdownV2 message to the configured chat. Telegram's MarkdownV2 requires
escaping a specific set of punctuation anywhere it appears in plain text —
we use a single regex for that and apply it to every user-facing field.

CLI: `python -m src.notifier --test` sends a one-shot hello ping so you can
verify the bot token and chat_id before wiring up the full pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import re

import structlog
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from src.config import Settings, get_settings
from src.models import JobPosting, JobScore

log = structlog.get_logger(__name__)

# MarkdownV2 reserved characters per https://core.telegram.org/bots/api#markdownv2-style
_MDV2_ESCAPE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _esc(text: str) -> str:
    """Escape every MarkdownV2-reserved char in `text`."""
    return _MDV2_ESCAPE.sub(r"\\\1", text or "")


def format_message(job: JobPosting, score: JobScore) -> str:
    """Render a single job+score pair as a MarkdownV2 Telegram message.

    Layout (emoji header → meta → strengths → red flags → link):
        🎯 Score 8/10 — Title @ Company
        📍 Location · source
        ✅ Strengths: ...
        ⚠️ Red flags: ...
        🔗 <url>
    """
    strengths = "; ".join(score.strengths[:3]) if score.strengths else "—"
    red_flags = "; ".join(score.red_flags[:3]) if score.red_flags else "none"

    lines = [
        f"🎯 *Score {score.fit_score}/10* — {_esc(job.title)} @ *{_esc(job.company)}*",
        f"📍 {_esc(job.location)} · _{_esc(job.source)}_",
        f"✅ {_esc(strengths)}",
        f"⚠️ {_esc(red_flags)}",
        f"🔗 {_esc(job.url)}",
    ]
    return "\n".join(lines)


async def send_job_match(bot: Bot, chat_id: str, job: JobPosting, score: JobScore) -> int | None:
    """Send one message. Returns Telegram message id, or None on failure."""
    text = format_message(job, score)
    try:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=False,
        )
        log.info("telegram_sent", job_id=job.id, message_id=msg.message_id, score=score.fit_score)
        return msg.message_id
    except TelegramError as e:
        log.warning("telegram_send_failed", job_id=job.id, error=str(e))
        return None


async def notify_matches(
    matches: list[tuple[JobPosting, JobScore]],
    *,
    settings: Settings | None = None,
) -> int:
    """Send a Telegram message for each (job, score). Returns count successfully sent."""
    if not matches:
        log.info("notify_skipped", reason="no_matches")
        return 0

    settings = settings or get_settings()
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    sent = 0
    for job, score in matches:
        mid = await send_job_match(bot, settings.telegram_chat_id, job, score)
        if mid is not None:
            sent += 1
    log.info("notify_done", matches=len(matches), sent=sent)
    return sent


async def _send_test_ping(settings: Settings) -> None:
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    text = _esc("Hello from job-hunter. Telegram wiring works.")
    await bot.send_message(
        chat_id=settings.telegram_chat_id,
        text=text,
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    log.info("test_ping_sent", chat_id=settings.telegram_chat_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram notifier utilities.")
    parser.add_argument("--test", action="store_true", help="Send a one-shot test ping.")
    args = parser.parse_args()

    if args.test:
        asyncio.run(_send_test_ping(get_settings()))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
