from typing import Literal

from pydantic import BaseModel, Field


class TopicSummary(BaseModel):
    name: str
    score: float = 0.0


class ResearcherSearchResult(BaseModel):
    researcher_id: str
    name: str
    works_count: int
    cited_by_count: int
    affiliation: str
    country_code: str | None = None
    topics: list[str] = Field(default_factory=list)


class ResearcherProfile(BaseModel):
    researcher_id: str
    name: str
    works_count: int
    cited_by_count: int
    h_index: int
    affiliation: str
    country_code: str | None = None
    homepage_url: str | None = None
    orcid: str | None = None
    topics: list[TopicSummary] = Field(default_factory=list)


class RecentWorkSummary(BaseModel):
    work_id: str
    title: str
    year: int | None = None
    cited_by_count: int = 0
    venue: str | None = None
    topics: list[str] = Field(default_factory=list)


class RecommendationExplanation(BaseModel):
    relevance: float
    diversity: float
    ecr_boost: float
    final_score: float
    reasons: list[str] = Field(default_factory=list)


class OpportunityRecommendation(BaseModel):
    opportunity_id: str
    opportunity_type: Literal["journal", "conference", "repository"]
    title: str
    score: float
    region: str
    deadline: str | None = None
    description: str
    homepage_url: str | None = None
    explanation: RecommendationExplanation


class CollaboratorRecommendation(BaseModel):
    researcher_id: str
    name: str
    affiliation: str
    country_code: str | None = None
    score: float
    explanation: RecommendationExplanation


class DashboardAnalytics(BaseModel):
    profile_strength: float
    cold_start_risk: float
    ecr_status: bool
    publication_count: int
    collaboration_readiness: float


class DashboardResponse(BaseModel):
    researcher: ResearcherProfile
    recent_works: list[RecentWorkSummary] = Field(default_factory=list)
    opportunities: list[OpportunityRecommendation]
    collaborators: list[CollaboratorRecommendation]
    analytics: DashboardAnalytics
