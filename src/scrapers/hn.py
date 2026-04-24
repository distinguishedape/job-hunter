"""Hacker News 'Who is hiring?' scraper.

Every first-of-the-month, user `whoishiring` posts the month's 'Who is hiring?'
thread. Top-level comments are individual job postings written in free-form
prose. We pull the latest thread via HN's public Firebase API (no anti-bot,
no auth needed), filter comments that mention Bengaluru/Bangalore/India *and*
an AI/ML/engineering signal, and return them as JobPosting records.

The comment text itself is messy (different posters use different formats), so
we hand the whole thing to the LLM scorer as the description — the LLM is
better than regex at parsing 'Company | Title | Location | Stack | Salary'
variations. We only do best-effort extraction of title/company here so the
database rows and Telegram messages look reasonable.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import structlog
from bs4 import BeautifulSoup

from src.models import JobPosting
from src.scrapers.base import Scraper, is_target_location

log = structlog.get_logger(__name__)

HN_API = "https://hacker-news.firebaseio.com/v0"
HN_USER = "whoishiring"
HN_WEB_ITEM = "https://news.ycombinator.com/item?id={id}"

# Extra keywords for HN - must match AI/ML/engineering signal *in the comment body*,
# not just the title (HN Who's-hiring comments have no structured title).
AI_SIGNAL = re.compile(
    r"\b(a\.?i\.?|ml|llm|machine[\s-]*learning|deep[\s-]*learning|nlp|"
    r"rag|agent|genai|gen-?ai|pytorch|tensorflow|anthropic|openai|claude|"
    r"data scientist|applied scientist|mle)\b",
    re.IGNORECASE,
)
ENGINEER_SIGNAL = re.compile(
    r"\b(engineer|developer|swe|sde|programmer|founding)\b",
    re.IGNORECASE,
)

# Concurrency limit when fetching hundreds of comments — HN tolerates this but
# we keep it reasonable to avoid hammering the free API
_CONCURRENCY = 20


class HNScraper(Scraper):
    name = "hn"

    def __init__(self, timeout_s: int = 20, concurrency: int = _CONCURRENCY) -> None:
        self.timeout_s = timeout_s
        self.concurrency = concurrency

    async def scrape(self) -> list[JobPosting]:
        log.info("scrape_start", source=self.name, url=f"{HN_API}/user/{HN_USER}.json")

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            thread_id, thread_title = await self._find_latest_whoishiring(client)
            if thread_id is None:
                log.warning("no_whoishiring_thread")
                return []

            log.info("thread_found", id=thread_id, title=thread_title)

            story = await self._get_item(client, thread_id)
            kid_ids = story.get("kids") or []
            log.info("fetching_comments", count=len(kid_ids))

            sem = asyncio.Semaphore(self.concurrency)

            async def fetch_one(cid: int) -> dict | None:
                async with sem:
                    return await self._get_item(client, cid)

            comments = await asyncio.gather(*(fetch_one(k) for k in kid_ids))

        jobs: list[JobPosting] = []
        for c in comments:
            if not c or c.get("dead") or c.get("deleted"):
                continue
            posting = self._comment_to_job(c, thread_title)
            if posting:
                jobs.append(posting)

        log.info(
            "scrape_done",
            source=self.name,
            total_comments=len(kid_ids),
            kept=len(jobs),
        )
        return jobs

    async def _find_latest_whoishiring(
        self, client: httpx.AsyncClient
    ) -> tuple[int | None, str]:
        """Walk whoishiring's submissions and return the latest 'Who is hiring?' story."""
        user = (await self._get_item_raw(client, f"/user/{HN_USER}.json")) or {}
        for sid in user.get("submitted", []):
            story = await self._get_item(client, sid)
            title = story.get("title", "")
            # Match 'Who is hiring?' but not 'Who wants to be hired?' or freelancer threads
            if (
                "hiring" in title.lower()
                and "wants to be hired" not in title.lower()
                and "freelancer" not in title.lower()
            ):
                return sid, title
        return None, ""

    async def _get_item(self, client: httpx.AsyncClient, item_id: int) -> dict:
        return (await self._get_item_raw(client, f"/item/{item_id}.json")) or {}

    @staticmethod
    async def _get_item_raw(client: httpx.AsyncClient, path: str) -> dict | None:
        resp = await client.get(f"{HN_API}{path}")
        resp.raise_for_status()
        return resp.json()

    def _comment_to_job(self, comment: dict, thread_title: str) -> JobPosting | None:
        text_html = comment.get("text") or ""
        if not text_html:
            return None

        soup = BeautifulSoup(text_html, "html.parser")
        # Replace <p> with newlines so paragraph breaks survive get_text()
        for p in soup.find_all("p"):
            p.insert_before("\n\n")
        text = soup.get_text().strip()

        # Filter: must mention Bengaluru/Bangalore/India + AI/ML + engineering signal
        if not is_target_location(text):
            return None
        if not (AI_SIGNAL.search(text) and ENGINEER_SIGNAL.search(text)):
            return None

        company, title = _parse_header(text)
        location = _extract_location(text)
        cid = comment["id"]

        return JobPosting(
            source=self.name,
            company=company,
            title=title,
            location=location,
            url=HN_WEB_ITEM.format(id=cid),
            description=f"{thread_title}\n\n{text}",
            posted_date=None,
        )


_HEADER_SEP = re.compile(r"\s*[|•·—–]\s*|\s+-\s+")


def _parse_header(text: str) -> tuple[str, str]:
    """Best-effort extraction of (company, title) from the first line of an HN comment.

    Most posters use one of these patterns:
        Company | Title | Location | Stack | Salary
        Company - Title - Location
        Company (YC Wxx) - Title
    When the heuristic fails we return ('HN Who is hiring', <first line>).
    """
    first_line = text.splitlines()[0].strip() if text else ""
    if not first_line:
        return "HN Who is hiring", "Job posting"

    parts = [p.strip() for p in _HEADER_SEP.split(first_line) if p.strip()]
    if len(parts) >= 2:
        company = parts[0][:80]
        title = parts[1][:120]
        return company, title

    return "HN Who is hiring", first_line[:120]


def _extract_location(text: str) -> str:
    """Return the first matched location phrase for display — full context stays in description."""
    for kw in ("Bengaluru", "Bangalore", "BLR", "India", "Remote"):
        if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
            return kw
    return "India"
