"""Internshala scraper.

Internshala is the largest Indian internship / fresher-jobs portal. Its search
pages render jobs server-side as static HTML with clean, stable selectors, and
it serves ordinary httpx requests with no anti-bot protection.

We hit 3 category URLs concurrently (AI / ML / Data Science, each filtered to
Bangalore), dedupe across the overlap by the `internshipid` attribute, and
filter each card through the standard role keywords as a safety net.

Card DOM (static HTML):

    <div class="individual_internship ..." internshipid="3109125"
         data-href="/internship/detail/...">
      <h2 class="job-internship-name"><a>{title}</a></h2>
      <p class="company-name">{company}</p>
      <div class="locations"><span><a>{location}</a></span></div>
      <span class="stipend">{stipend}</span>
      <div class="about_job"><div class="text">{description}</div></div>
      <div class="job_skills">…skill tags…</div>
    </div>
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from bs4 import BeautifulSoup, Tag

from src.models import JobPosting
from src.scrapers.base import Scraper, is_target_role

log = structlog.get_logger(__name__)

INTERNSHALA_BASE = "https://internshala.com"
# Category paths we hit concurrently; overlap is expected and deduped by internshipid
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "artificial-intelligence-internship-in-bangalore",
    "machine-learning-internship-in-bangalore",
    "data-science-internship-in-bangalore",
)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class InternshalaScraper(Scraper):
    name = "internshala"

    def __init__(
        self,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        timeout_s: int = 30,
    ) -> None:
        self.categories = categories
        self.timeout_s = timeout_s

    async def scrape(self) -> list[JobPosting]:
        log.info("scrape_start", source=self.name, categories=list(self.categories))

        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            pages = await asyncio.gather(
                *(self._fetch(client, cat) for cat in self.categories),
                return_exceptions=True,
            )

        seen_ids: set[str] = set()
        jobs: list[JobPosting] = []
        total_cards = 0

        for cat, page in zip(self.categories, pages, strict=True):
            if isinstance(page, BaseException):
                log.warning("category_failed", category=cat, error=str(page))
                continue

            cards = self._parse_cards(page)
            total_cards += len(cards)

            for card in cards:
                if card["internship_id"] in seen_ids:
                    continue
                if not is_target_role(card["title"]):
                    continue
                seen_ids.add(card["internship_id"])
                jobs.append(
                    JobPosting(
                        source=self.name,
                        company=card["company"],
                        title=card["title"],
                        location=card["location"],
                        url=card["url"],
                        description=card["description"],
                        posted_date=None,  # Internshala doesn't publish a date on list cards
                    )
                )

        log.info(
            "scrape_done",
            source=self.name,
            total_cards=total_cards,
            unique_kept=len(jobs),
        )
        return jobs

    async def _fetch(self, client: httpx.AsyncClient, category: str) -> str:
        url = f"{INTERNSHALA_BASE}/internships/{category}"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text

    def _parse_cards(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        cards: list[dict] = []

        for card in soup.select(".individual_internship"):
            internship_id = card.get("internshipid") or ""
            if not internship_id:
                continue

            title_el = card.select_one(".job-internship-name a")
            company_el = card.select_one(".company-name")
            location_el = card.select_one(".locations span a")
            description_el = card.select_one(".about_job .text")
            stipend_el = card.select_one(".stipend")
            skills_el = card.select(".job_skills .round_tabs, .round_tabs")

            data_href = card.get("data-href") or (title_el.get("href") if title_el else "")
            url = f"{INTERNSHALA_BASE}{data_href}" if data_href and str(data_href).startswith("/") else str(data_href or "")

            description = _text(description_el)
            if stipend_el:
                description = f"Stipend: {_text(stipend_el)}\n\n{description}"
            skills = [_text(s) for s in skills_el if _text(s)]
            if skills:
                description = f"{description}\n\nSkills: {', '.join(skills)}"

            cards.append(
                {
                    "internship_id": str(internship_id),
                    "title": _text(title_el),
                    "company": _text(company_el),
                    "location": _text(location_el) or "Bangalore",  # category URL pre-filters to BLR
                    "url": url,
                    "description": description,
                }
            )

        return cards


def _text(el: Tag | None) -> str:
    if el is None:
        return ""
    return el.get_text(" ", strip=True)
