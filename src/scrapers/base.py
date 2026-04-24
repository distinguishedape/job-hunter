"""Abstract scraper interface.

Each scraper is an async callable that returns a list of JobPosting objects.
Location + role filtering happens inside the scraper so callers don't need
to know the quirks of each source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from src.models import JobPosting

log = structlog.get_logger(__name__)

# Keywords we treat as Bengaluru-adjacent / India-remote-friendly
LOCATION_KEYWORDS = (
    "bengaluru",
    "bangalore",
    "blr",
    "india",
    "remote (india)",
    "remote - india",
)

# Keywords for engineer-ish / AI-ish roles
ROLE_KEYWORDS = (
    "engineer",
    "developer",
    "sde",
    "swe",
    "programmer",
    "ml ",
    "ai ",
    " ai",
    "machine learning",
    "data scientist",
    "research",
    "founding",
    "applied scientist",
    "mle",
    "llm",
)


def is_target_location(location: str) -> bool:
    if not location:
        return False
    loc = location.lower()
    return any(k in loc for k in LOCATION_KEYWORDS)


def is_target_role(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(k in t for k in ROLE_KEYWORDS)


class Scraper(ABC):
    """Base class for all job-board scrapers."""

    name: str = "base"

    @abstractmethod
    async def scrape(self) -> list[JobPosting]:
        """Fetch jobs from the source and return filtered postings."""
