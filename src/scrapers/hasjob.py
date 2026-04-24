"""Hasjob.co (Hasgeek) scraper.

Hasjob is the community job board run by Hasgeek — the Bengaluru tech collective
behind PyCon India, The Fifth Elephant, and Rootconf. It publishes a public
Atom feed at /feed with ~14 current listings, each tagged with a structured
<location> element. No Cloudflare, no login required.

Swapped in for Wellfound (which is hard-blocked by Cloudflare Turnstile from
every non-residential IP).  See README for full rationale.
"""

from __future__ import annotations

import html
from datetime import datetime
from xml.etree import ElementTree as ET

import httpx
import structlog
from bs4 import BeautifulSoup

from src.models import JobPosting
from src.scrapers.base import Scraper, is_target_location, is_target_role

log = structlog.get_logger(__name__)

HASJOB_FEED_URL = "https://hasjob.co/feed"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class HasjobScraper(Scraper):
    name = "hasjob"

    def __init__(self, timeout_s: int = 30) -> None:
        self.timeout_s = timeout_s

    async def scrape(self) -> list[JobPosting]:
        log.info("scrape_start", source=self.name, url=HASJOB_FEED_URL)

        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            resp = await client.get(
                HASJOB_FEED_URL,
                headers={"User-Agent": "Mozilla/5.0 (job-hunter/0.1)"},
            )
            resp.raise_for_status()
            feed_xml = resp.text

        raw = self._parse_feed(feed_xml)

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
            total_entries=len(raw),
            kept=len(jobs),
        )
        return jobs

    def _parse_feed(self, feed_xml: str) -> list[dict]:
        """Parse Atom XML into raw dicts. Each <entry> gives us everything we need."""
        root = ET.fromstring(feed_xml)
        entries: list[dict] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            title = _text(entry.find("atom:title", ATOM_NS))
            link_el = entry.find("atom:link", ATOM_NS)
            url = link_el.get("href", "") if link_el is not None else ""
            published = _text(entry.find("atom:published", ATOM_NS))
            # Hasjob's <location> element inherits the default Atom xmlns, so it
            # has to be looked up with the atom: prefix even though it's a custom tag
            location = _text(entry.find("atom:location", ATOM_NS))

            content_raw = _text(entry.find("atom:content", ATOM_NS))
            # Atom content is HTML-encoded; decode entities then strip tags for description
            content_html = html.unescape(content_raw)
            soup = BeautifulSoup(content_html, "html.parser")

            # First <strong><a> inside content is the company name + link
            first_anchor = soup.find("a")
            company = (
                first_anchor.get_text(strip=True)
                if first_anchor and first_anchor.get_text(strip=True)
                else _company_from_url(url)
            )

            description = soup.get_text(separator="\n", strip=True)

            entries.append(
                {
                    "title": title,
                    "url": url,
                    "company": company,
                    "location": location,
                    "description": description,
                    "posted_date": _normalize_date(published),
                }
            )

        return entries


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _company_from_url(url: str) -> str:
    """Fallback when the feed entry has no company anchor — derive from hasjob.co/<company>/<id>."""
    # https://hasjob.co/exceleron.com/kqqs4 -> "exceleron.com"
    parts = [p for p in url.split("/") if p]
    return parts[-2] if len(parts) >= 2 else "unknown"


def _normalize_date(iso_ts: str) -> str:
    """Convert 2026-04-22T10:00:22.252377+00:00 -> 2026-04-22."""
    if not iso_ts:
        return ""
    try:
        return datetime.fromisoformat(iso_ts).date().isoformat()
    except ValueError:
        return iso_ts[:10]
