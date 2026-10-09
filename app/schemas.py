from typing import Literal

from pydantic import BaseModel, Field


CollaboratorScope = Literal["global", "same_country", "same_affiliation"]
EmbeddingModelName = Literal["hashing", "scibert", "specter2"]


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
    landing_page_url: str | None = None
    source_type: str | None = None
    topics: list[str] = Field(default_factory=list)


class MLModelScore(BaseModel):
    name: str
    score: float
    role: str


class RecommendationExplanation(BaseModel):
    relevance: float
    diversity: float
    ecr_boost: float
    final_score: float
    reasons: list[str] = Field(default_factory=list)
    model_scores: list[MLModelScore] = Field(default_factory=list)


class OpportunityRecommendation(BaseModel):
    opportunity_id: str
    opportunity_type: Literal["journal", "conference", "repository", "paper"]
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
    scope_match: str = "global"
    evidence_works: list[RecentWorkSummary] = Field(default_factory=list)
    explanation: RecommendationExplanation


class DashboardAnalytics(BaseModel):
    profile_strength: float
    cold_start_risk: float
    ecr_status: bool
    publication_count: int
    collaboration_readiness: float


class PromotionAction(BaseModel):
    title: str
    category: Literal["profile", "venue", "collaboration", "visibility", "publication"]
    priority: Literal["high", "medium", "low"]
    rationale: str
    expected_impact: float
    status: Literal["recommended", "in_progress", "completed"] = "recommended"


class VisibilityGapDimension(BaseModel):
    name: str
    current: float
    target: float
    gap: float
    explanation: str


class PromotionRoadmapStep(BaseModel):
    phase: str
    title: str
    objective: str
    metric: str


class MLSignal(BaseModel):
    name: str
    value: str
    description: str


class PromotionTracker(BaseModel):
    promotion_score: float
    visibility_gap: float
    citation_growth_potential: float
    collaboration_growth_potential: float
    visibility_twin_summary: str
    gap_dimensions: list[VisibilityGapDimension] = Field(default_factory=list)
    roadmap: list[PromotionRoadmapStep] = Field(default_factory=list)
    ml_signals: list[MLSignal] = Field(default_factory=list)
    recommended_actions: list[PromotionAction] = Field(default_factory=list)
    next_review_window: str


class DashboardResponse(BaseModel):
    researcher: ResearcherProfile
    recent_works: list[RecentWorkSummary] = Field(default_factory=list)
    opportunities: list[OpportunityRecommendation]
    recommended_works: list[OpportunityRecommendation] = Field(default_factory=list)
    collaborators: list[CollaboratorRecommendation]
    analytics: DashboardAnalytics
    collaborator_scope: CollaboratorScope = "global"
    embedding_model: EmbeddingModelName = "hashing"
    embedding_model_status: str = "Local hashing baseline active."
    collaborator_model_status: str = "GraphSAGE embeddings are not loaded."


class ConnectedProfile(BaseModel):
    platform: str
    url: str
    status: Literal["connected", "missing"]
    note: str


class RecommendationSummary(BaseModel):
    mode: Literal["orcid_openalex", "cold_start"]
    profile_basis: str
    recommendation_targets: list[str] = Field(default_factory=list)
    explanation_steps: list[str] = Field(default_factory=list)


class UserCabinetResponse(BaseModel):
    registered_profile: "RegisteredResearcherProfile"
    personalized_dashboard: DashboardResponse
    connected_profiles: list[ConnectedProfile] = Field(default_factory=list)
    verified_projects: list[str] = Field(default_factory=list)
    recommendation_summary: RecommendationSummary
    promotion_tracker: PromotionTracker


class DomainCandidate(BaseModel):
    name: str
    score: float
    matched_terms: list[str] = Field(default_factory=list)


class ResearcherRegistrationRequest(BaseModel):
    full_name: str
    email: str
    orcid_id: str | None = None
    affiliation: str = ""
    position_title: str = ""
    academic_level: str = ""
    academic_field: str = ""
    department: str | None = None
    country: str = ""
    city: str | None = None
    years_in_research: int = 0
    publication_count: int = 0
    citation_count: int = 0
    h_index: int = 0
    i10_index: int = 0
    conference_participation_count: int = 0
    peer_review_count: int = 0
    grant_count: int = 0
    project_count: int = 0
    collaboration_count: int = 0
    profile_views: int = 0
    downloads: int = 0
    altmetric_mentions: int = 0
    research_keywords: list[str] = Field(default_factory=list)
    thematic_clusters: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    target_journals: list[str] = Field(default_factory=list)
    collaboration_goals: list[str] = Field(default_factory=list)
    publication_goals: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    preferred_languages: list[str] = Field(default_factory=list)
    google_scholar_url: str | None = None
    researchgate_url: str | None = None
    personal_website_url: str | None = None
    bio: str | None = None


class AccountRegistrationRequest(ResearcherRegistrationRequest):
    password: str = Field(min_length=8)


class RegisteredResearcherProfile(BaseModel):
    profile_id: str
    created_at: str
    full_name: str
    email: str
    affiliation: str
    position_title: str
    academic_level: str
    academic_field: str
    country: str
    profile_completeness: float
    visibility_score: float
    research_keywords: list[str] = Field(default_factory=list)
    thematic_clusters: list[str] = Field(default_factory=list)
    interest_embedding_text: str
    domain_candidates: list[DomainCandidate] = Field(default_factory=list)
    derived_metrics: dict[str, int | str] = Field(default_factory=dict)
    profile_data_sources: list[str] = Field(default_factory=list)
    raw_profile: ResearcherRegistrationRequest


class UserLoginRequest(BaseModel):
    email: str
    password: str


class AuthenticatedUser(BaseModel):
    user_id: str
    email: str
    profile_id: str
    created_at: str
    last_login_at: str | None = None


class AuthSessionResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: AuthenticatedUser
    profile: RegisteredResearcherProfile


class AuthenticatedCabinetResponse(BaseModel):
    user: AuthenticatedUser
    cabinet: UserCabinetResponse


class PromotionSnapshot(BaseModel):
    snapshot_id: str
    profile_id: str
    created_at: str
    promotion_score: float
    visibility_score: float
    works_count: int
    citation_count: int
    h_index: int
    collaborator_count: int
    completed_actions: int = 0


class PromotionActionUpdate(BaseModel):
    title: str
    status: Literal["recommended", "in_progress", "completed"]


class PromotionProgressResponse(BaseModel):
    profile_id: str
    baseline: PromotionSnapshot | None = None
    latest: PromotionSnapshot | None = None
    deltas: dict[str, float | int] = Field(default_factory=dict)
    snapshots: list[PromotionSnapshot] = Field(default_factory=list)


class ExperimentPlanResponse(BaseModel):
    dataset_source: str
    target_slice: str
    models: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    validation_strategy: str
    notes: list[str] = Field(default_factory=list)


UserCabinetResponse.model_rebuild()
AuthSessionResponse.model_rebuild()
AuthenticatedCabinetResponse.model_rebuild()
