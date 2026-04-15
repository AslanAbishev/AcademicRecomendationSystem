from typing import Literal

from pydantic import BaseModel, Field


class ResearcherSummary(BaseModel):
    researcher_id: int
    name: str
    phd_year: int
    domain: str
    region: str
    career_stage: str
    h_index: int
    publication_count: int


class RecommendationExplanation(BaseModel):
    relevance: float
    diversity: float
    ecr_boost: float
    final_score: float
    reasons: list[str] = Field(default_factory=list)


class OpportunityRecommendation(BaseModel):
    opportunity_id: str
    opportunity_type: Literal["journal", "conference", "grant", "mentor"]
    title: str
    score: float
    region: str
    deadline: str
    explanation: RecommendationExplanation


class CollaboratorRecommendation(BaseModel):
    researcher_id: int
    name: str
    domain: str
    score: float
    explanation: RecommendationExplanation


class DashboardAnalytics(BaseModel):
    profile_strength: float
    cold_start_risk: float
    ecr_status: bool
    publication_count: int
    collaboration_readiness: float


class DashboardResponse(BaseModel):
    researcher: ResearcherSummary
    opportunities: list[OpportunityRecommendation]
    collaborators: list[CollaboratorRecommendation]
    analytics: DashboardAnalytics
