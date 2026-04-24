"""Y Combinator jobs scraper.

Scrapes https://www.ycombinator.com/jobs (a static-rendered list of ~20 latest
postings) and filters for Bengaluru/India engineering roles. The page has no
server-side location filter, so filtering happens in Python.

Each job card is an <li> with this rough shape:

    <li>
      <a href="/companies/{slug}">…</a>           # company
      <span>{Company} (W24)</span>                # batch badge
      <span>{tagline}</span>
      <a href="/companies/{slug}/jobs/{id}-{slug}">{Title}</a>
      <div class="flex …">
        <div>Full-time</div> • <div>Engineering</div> • … • <div>{Location}</div>
      </div>
    </li>
"""

from __future__ import annotations

import structlog
from playwright.async_api import async_playwright

from src.models import JobPosting
from src.scrapers.base import Scraper, is_target_location, is_target_role

log = structlog.get_logger(__name__)

YC_JOBS_URL = "https://www.ycombinator.com/jobs"
YC_BASE = "https://www.ycombinator.com"


class YCScraper(Scraper):
    name = "yc"

    def __init__(self, headless: bool = True, timeout_ms: int = 60_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms

    async def scrape(self) -> list[JobPosting]:
        log.info("scrape_start", source=self.name, url=YC_JOBS_URL)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page()
                await page.goto(YC_JOBS_URL, wait_until="networkidle", timeout=self.timeout_ms)
                raw = await self._extract_cards(page)
            finally:
                await browser.close()

        jobs: list[JobPosting] = []
        for r in raw:
            if not is_target_location(r["location"]) or not is_target_role(r["title"]):
                continue
            jobs.append(
                JobPosting(
                    source=self.name,
                    company=r["company"],
                    title=r["title"],
                    location=r["location"],
                    url=r["url"],
                    description=r["description"],
                    posted_date=r.get("posted_date"),
                )
            )

        log.info(
            "scrape_done",
            source=self.name,
            total_cards=len(raw),
            kept=len(jobs),
        )
        return jobs

    async def _extract_cards(self, page) -> list[dict]:  # type: ignore[no-untyped-def]
        """Pull every job <li> on the page into a list of raw dicts."""
        return await page.evaluate(
            """
            () => {
                const cards = Array.from(document.querySelectorAll('li'))
                    .filter(li => li.querySelector('a[href*="/jobs/"]'));

                return cards.map(li => {
                    const titleLink = li.querySelector('a[href*="/jobs/"]');
                    const companyLink = li.querySelector('a[href^="/companies/"]:not([href*="/jobs/"])');
                    const batchSpan = li.querySelector('span.font-bold');
                    const taglineSpan = li.querySelector('span.text-gray-700');

                    // The last flex-wrap tag row holds Full-time • dept • specialty • salary • location
                    const tagRow = li.querySelector('.flex.flex-wrap');
                    const tags = tagRow
                        ? Array.from(tagRow.querySelectorAll('div'))
                            .map(d => d.innerText.trim())
                            .filter(t => t && t !== '•' && t.length > 1)
                        : [];
                    const location = tags.length ? tags[tags.length - 1] : '';

                    // "about 15 hours ago" style relative timestamp
                    const tsSpan = li.querySelector('span.text-gray-400');
                    const postedRelative = tsSpan ? tsSpan.innerText.replace(/[()]/g, '').trim() : '';

                    const tagline = taglineSpan ? taglineSpan.innerText.trim() : '';
                    const batch = batchSpan ? batchSpan.innerText.trim() : '';

                    return {
                        title: titleLink ? titleLink.innerText.trim() : '',
                        url: titleLink ? titleLink.href : '',
                        company: batch.replace(/\\s*\\([WS]\\d+\\)\\s*$/, '').trim(),
                        company_batch: batch,
                        tags,
                        location,
                        posted_date: postedRelative,
                        description: [batch, tagline, tags.join(' • ')].filter(Boolean).join('\\n'),
                    };
                });
            }
            """
        )
