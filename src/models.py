"""Pydantic models for job postings and LLM-generated scores."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class JobPosting(BaseModel):
    source: str
    company: str
    title: str
    location: str
    url: str
    description: str
    posted_date: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field  # type: ignore[misc]
    @property
    def id(self) -> str:
        return hashlib.sha256(f"{self.url}|{self.title}".encode()).hexdigest()[:16]

    model_config = {"populate_by_name": True}


class JobScore(BaseModel):
    job_id: str
    fit_score: int = Field(ge=1, le=10, description="Overall fit score from 1 (poor) to 10 (perfect)")
    reasons: list[str] = Field(
        min_length=2,
        max_length=4,
        description="2-4 concise reasons explaining the score",
    )
    strengths: list[str] = Field(description="Specific matching strengths between the role and the CV")
    red_flags: list[str] = Field(description="Concerns or mismatches to be aware of")
    should_apply: bool = Field(description="True if fit_score >= 7 and no disqualifying red flags")
