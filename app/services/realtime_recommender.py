from __future__ import annotations

from collections import defaultdict
import json
import re

import numpy as np

from app.clients.openalex import OpenAlexClient
from app.config import settings
from app.schemas import (
    CollaboratorRecommendation,
    CollaboratorScope,
    ConnectedProfile,
    DashboardAnalytics,
    DashboardResponse,
    EmbeddingModelName,
    MLModelScore,
    MLSignal,
    OpportunityRecommendation,
    PromotionAction,
    PromotionRoadmapStep,
    PromotionTracker,
    RecentWorkSummary,
    RecommendationExplanation,
    RecommendationSummary,
    RegisteredResearcherProfile,
    ResearcherProfile,
    ResearcherSearchResult,
    TopicSummary,
    UserCabinetResponse,
    VisibilityGapDimension,
)
from app.services.ml_models import HybridSemanticScorer, HybridSimilarityScore
from app.services.graphsage_ranker import GraphSAGERanker
from app.services.research_database import ResearchDatabase
from app.services.scientific_embeddings import ScientificEmbeddingModel


CURRENT_YEAR = 2026
MIN_COLLABORATOR_RELEVANCE = 0.03

COUNTRY_ALIASES = {
    "kazakhstan": "KZ",
    "kz": "KZ",
    "germany": "DE",
    "de": "DE",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "china": "CN",
    "russia": "RU",
    "russian federation": "RU",
    "turkey": "TR",
    "turkiye": "TR",
    "canada": "CA",
    "netherlands": "NL",
    "france": "FR",
    "spain": "ES",
    "italy": "IT",
}


def _safe_float(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize_affiliation(value: str | None) -> str:
    if not value or value == "Unknown affiliation":
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value.lower()).split())


def _affiliation_tokens(value: str | None) -> set[str]:
    stop_words = {
        "of",
        "the",
        "and",
        "university",
        "universitet",
        "institute",
        "college",
        "school",
        "faculty",
        "department",
        "national",
        "state",
    }
    return {
        token
        for token in _normalize_affiliation(value).split()
        if len(token) > 2 and token not in stop_words
    }


def _affiliations_match(left: str | None, right: str | None) -> bool:
    normalized_left = _normalize_affiliation(left)
    normalized_right = _normalize_affiliation(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    left_tokens = _affiliation_tokens(left)
    right_tokens = _affiliation_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) / min(len(left_tokens), len(right_tokens)) >= 0.6


def _normalize_country(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    if len(cleaned) == 2:
        return cleaned.upper()
    return COUNTRY_ALIASES.get(cleaned.lower(), cleaned.upper())


def _short_openalex_id(value: str | None) -> str:
    return value.rstrip("/").split("/")[-1] if value else ""


def _extract_affiliation(author_payload: dict) -> str:
    institution = author_payload.get("last_known_institutions") or []
    if institution:
        return institution[0].get("display_name", "Unknown affiliation")
    return "Unknown affiliation"


def _extract_country(author_payload: dict) -> str | None:
    institution = author_payload.get("last_known_institutions") or []
    if institution:
        return institution[0].get("country_code")
    return None


def _extract_affiliation_from_authorship(authorship: dict) -> str:
    institutions = authorship.get("institutions") or []
    if institutions:
        return institutions[0].get("display_name", "Unknown affiliation")
    return "Unknown affiliation"


def _extract_country_from_authorship(authorship: dict) -> str | None:
    countries = authorship.get("countries") or []
    if countries:
        return countries[0]
    institutions = authorship.get("institutions") or []
    if institutions:
        return institutions[0].get("country_code")
    return None


def _topic_names(payload_topics: list[dict], limit: int = 5) -> list[str]:
    return [item.get("display_name", "") for item in payload_topics[:limit] if item.get("display_name")]


def _topic_ids(payload_topics: list[dict], limit: int = 5) -> list[str]:
    return [item.get("id", "") for item in payload_topics[:limit] if item.get("id")]


def _topic_summaries(payload_topics: list[dict], limit: int = 8) -> list[TopicSummary]:
    return [
        TopicSummary(name=item.get("display_name", ""), score=float(item.get("count", 0)))
        for item in payload_topics[:limit]
        if item.get("display_name")
    ]


class ECRRecommenderService:
    def __init__(self, client: OpenAlexClient | None = None, research_db: ResearchDatabase | None = None) -> None:
        self.client = client or OpenAlexClient(
            mailto=settings.openalex_mailto,
            api_key=settings.openalex_api_key,
        )
        self.scorer = HybridSemanticScorer()
        self.research_db = research_db or ResearchDatabase(settings.database_url)
        self._embedding_models: dict[str, ScientificEmbeddingModel] = {}
        self._graphsage_ranker = GraphSAGERanker(
            settings.graphsage_embeddings_path,
            settings.graphsage_model_path,
        )
        self._graphsage_status = self._graphsage_ranker.status
        self._collaborator_semantic_status = "Transformer collaborator ranking is not loaded."

    def search_researchers(self, query: str, limit: int = 8) -> list[ResearcherSearchResult]:
        results = self.client.search_authors(query, per_page=limit)
        return [self._map_search_result(item) for item in results]

    def get_dashboard(
        self,
        researcher_id: str,
        top_k: int = 6,
        collaborator_scope: CollaboratorScope = "global",
        embedding_model: EmbeddingModelName = "hashing",
        reference_affiliation: str | None = None,
        reference_country: str | None = None,
        profile_text_override: str | None = None,
    ) -> DashboardResponse:
        author = self.client.get_author(researcher_id)
        works = self.client.get_author_works(researcher_id, per_page=12)
        profile = self._map_profile(author)
        recent_works = [self._map_work_summary(work) for work in works]
        author_text = self._build_author_text(author, works)
        researcher_text = " ".join([profile_text_override or "", author_text]).strip()
        opportunities = self._recommend_sources(author, works, researcher_text, top_k=top_k)
        recommended_works, embedding_status = self._recommend_scientific_works(
            researcher_text,
            embedding_model=embedding_model,
            top_k=top_k,
            exclude_work_ids=[work.work_id for work in recent_works],
        )
        collaborators = self._recommend_collaborators(
            author,
            works,
            researcher_text,
            top_k=min(top_k, 5),
            collaborator_scope=collaborator_scope,
            reference_affiliation=reference_affiliation,
            reference_country=reference_country,
            embedding_model=embedding_model,
            current_works=works,
        )
        analytics = self._build_analytics(author, works)
        return DashboardResponse(
            researcher=profile,
            recent_works=recent_works,
            opportunities=opportunities,
            recommended_works=recommended_works,
            collaborators=collaborators,
            analytics=analytics,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
            embedding_model_status=embedding_status,
            collaborator_model_status=self._collaborator_model_status(),
        )

    def get_registered_profile_dashboard(
        self,
        profile: RegisteredResearcherProfile,
        top_k: int = 6,
        collaborator_scope: CollaboratorScope = "global",
        embedding_model: EmbeddingModelName = "hashing",
    ) -> DashboardResponse:
        researcher_text = self._build_registered_profile_text(profile)
        queries = self._build_registered_queries(profile)
        orcid_id = profile.raw_profile.orcid_id
        if orcid_id:
            try:
                author = self.client.get_author_by_orcid(orcid_id)
                author_id = author["id"].split("/")[-1]
                works = self.client.get_author_works(author_id, per_page=12)
                author_text = self._build_registered_author_text(profile, author, works)
                enriched_text = " ".join([researcher_text, author_text]).strip()
                enriched_queries = self._merge_queries([*queries, *self._build_topic_queries(author, works)])
                opportunities = self._recommend_sources_from_queries(enriched_queries, enriched_text, top_k=top_k)
                recent_works = [self._map_work_summary(work) for work in works]
                recommended_works, embedding_status = self._recommend_scientific_works(
                    enriched_text,
                    embedding_model=embedding_model,
                    top_k=top_k,
                    exclude_work_ids=[work.work_id for work in recent_works],
                )
                collaborators = self._recommend_collaborators_from_queries(
                    enriched_queries,
                    enriched_text,
                    top_k=min(top_k, 5),
                    current_author_id=author_id,
                    current_author=author,
                    collaborator_scope=collaborator_scope,
                    reference_affiliation=profile.affiliation,
                    reference_country=profile.country,
                    embedding_model=embedding_model,
                    current_works=works,
                )
                return DashboardResponse(
                    researcher=self._map_enriched_registered_profile(profile, author),
                    recent_works=recent_works,
                    opportunities=opportunities,
                    recommended_works=recommended_works,
                    collaborators=collaborators,
                    analytics=self._build_analytics(author, works),
                    collaborator_scope=collaborator_scope,
                    embedding_model=embedding_model,
                    embedding_model_status=embedding_status,
                    collaborator_model_status=self._collaborator_model_status(),
                )
            except Exception:
                pass

        opportunities = self._recommend_sources_from_queries(queries, researcher_text, top_k=top_k)
        recommended_works, embedding_status = self._recommend_scientific_works(
            researcher_text,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        collaborators = self._recommend_collaborators_from_queries(
            queries,
            researcher_text,
            top_k=min(top_k, 5),
            current_author_id=None,
            collaborator_scope=collaborator_scope,
            reference_affiliation=profile.affiliation,
            reference_country=profile.country,
            embedding_model=embedding_model,
        )
        return DashboardResponse(
            researcher=self._map_registered_profile(profile),
            recent_works=[],
            opportunities=opportunities,
            recommended_works=recommended_works,
            collaborators=collaborators,
            analytics=self._build_registered_analytics(profile),
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
            embedding_model_status=embedding_status,
            collaborator_model_status=self._collaborator_model_status(),
        )

    def get_user_cabinet(
        self,
        profile: RegisteredResearcherProfile,
        top_k: int = 6,
        collaborator_scope: CollaboratorScope = "global",
        embedding_model: EmbeddingModelName = "hashing",
    ) -> UserCabinetResponse:
        dashboard = self.get_registered_profile_dashboard(
            profile,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        )
        mode = "orcid_openalex" if profile.raw_profile.orcid_id and dashboard.recent_works else "cold_start"
        return UserCabinetResponse(
            registered_profile=profile,
            personalized_dashboard=dashboard,
            connected_profiles=self._build_connected_profiles(profile),
            verified_projects=[],
            recommendation_summary=RecommendationSummary(
                mode=mode,
                profile_basis=self._build_profile_basis(profile, mode),
                recommendation_targets=["venues", "collaborators", "scholarly visibility"],
                explanation_steps=self._build_recommendation_steps(mode),
            ),
            promotion_tracker=self._build_promotion_tracker(profile, dashboard),
        )

    def _map_search_result(self, author: dict) -> ResearcherSearchResult:
        return ResearcherSearchResult(
            researcher_id=author["id"].split("/")[-1],
            name=author.get("display_name", "Unknown author"),
            works_count=int(author.get("works_count", 0)),
            cited_by_count=int(author.get("cited_by_count", 0)),
            affiliation=_extract_affiliation(author),
            country_code=_extract_country(author),
            topics=_topic_names(author.get("topics", [])),
        )

    def _map_profile(self, author: dict) -> ResearcherProfile:
        ids = author.get("ids", {})
        summary_stats = author.get("summary_stats") or {}
        return ResearcherProfile(
            researcher_id=author["id"].split("/")[-1],
            name=author.get("display_name", "Unknown author"),
            works_count=int(author.get("works_count", 0)),
            cited_by_count=int(author.get("cited_by_count", 0)),
            h_index=int(summary_stats.get("h_index", 0)),
            affiliation=_extract_affiliation(author),
            country_code=_extract_country(author),
            homepage_url=author.get("homepage_url"),
            orcid=ids.get("orcid"),
            topics=_topic_summaries(author.get("topics", [])),
        )

    def _map_enriched_registered_profile(self, profile: RegisteredResearcherProfile, author: dict) -> ResearcherProfile:
        ids = author.get("ids", {})
        summary_stats = author.get("summary_stats") or {}
        return ResearcherProfile(
            researcher_id=author["id"].split("/")[-1],
            name=profile.full_name,
            works_count=int(author.get("works_count", profile.derived_metrics.get("works_count", 0))),
            cited_by_count=int(author.get("cited_by_count", profile.derived_metrics.get("citation_count", 0))),
            h_index=int(summary_stats.get("h_index", profile.derived_metrics.get("h_index", 0))),
            affiliation=profile.affiliation,
            country_code=_normalize_country(profile.country) or _extract_country(author),
            homepage_url=profile.raw_profile.personal_website_url or author.get("homepage_url"),
            orcid=profile.raw_profile.orcid_id or ids.get("orcid"),
            topics=self._merge_profile_topics(profile, author),
        )

    def _merge_profile_topics(self, profile: RegisteredResearcherProfile, author: dict) -> list[TopicSummary]:
        topics: list[TopicSummary] = []
        seen: set[str] = set()
        for item in profile.domain_candidates:
            key = item.name.lower()
            if key not in seen:
                seen.add(key)
                topics.append(TopicSummary(name=item.name, score=item.score))
        for item in _topic_summaries(author.get("topics", []), limit=8):
            key = item.name.lower()
            if key not in seen:
                seen.add(key)
                topics.append(item)
        return topics[:8]

    def _map_work_summary(self, work: dict) -> RecentWorkSummary:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        return RecentWorkSummary(
            work_id=work["id"].split("/")[-1],
            title=work.get("title", "Untitled work"),
            year=work.get("publication_year"),
            cited_by_count=int(work.get("cited_by_count", 0)),
            venue=source.get("display_name"),
            landing_page_url=work.get("doi") or work.get("primary_location", {}).get("landing_page_url"),
            source_type=source.get("type"),
            topics=_topic_names(work.get("topics", []), limit=4),
        )

    def _build_author_text(self, author: dict, works: list[dict]) -> str:
        text_parts = [
            author.get("display_name", ""),
            _extract_affiliation(author),
            " ".join(_topic_names(author.get("topics", []), limit=8)),
        ]
        for work in works[:8]:
            text_parts.append(work.get("title", ""))
            text_parts.append(" ".join(_topic_names(work.get("topics", []), limit=4)))
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            text_parts.append(source.get("display_name", ""))
        return " ".join(part for part in text_parts if part).strip()

    def _recommend_sources(
        self,
        author: dict,
        works: list[dict],
        researcher_text: str,
        top_k: int,
    ) -> list[OpportunityRecommendation]:
        queries = self._build_topic_queries(author, works)
        return self._recommend_sources_from_queries(queries, researcher_text, top_k=top_k)

    def _recommend_collaborators(
        self,
        author: dict,
        works: list[dict],
        researcher_text: str,
        top_k: int,
        collaborator_scope: CollaboratorScope,
        reference_affiliation: str | None = None,
        reference_country: str | None = None,
        embedding_model: EmbeddingModelName = "specter2",
        current_works: list[dict] | None = None,
    ) -> list[CollaboratorRecommendation]:
        queries = self._build_topic_queries(author, works)
        return self._recommend_collaborators_from_queries(
            queries,
            researcher_text,
            top_k=top_k,
            current_author_id=author["id"].split("/")[-1],
            current_author=author,
            collaborator_scope=collaborator_scope,
            reference_affiliation=reference_affiliation,
            reference_country=reference_country,
            embedding_model=embedding_model,
            current_works=works,
        )

    def _recommend_sources_from_queries(
        self,
        queries: list[str],
        researcher_text: str,
        top_k: int,
    ) -> list[OpportunityRecommendation]:
        source_candidates: dict[str, dict] = {}

        for query in queries:
            for source in self.client.search_sources(query, per_page=6):
                source_candidates[source["id"]] = source

        if not source_candidates:
            return []

        candidate_values = list(source_candidates.values())
        texts = [self._build_source_text(item) for item in candidate_values]
        similarity_scores = self.scorer.score(researcher_text, texts)

        recommendations: list[OpportunityRecommendation] = []
        seen_types: defaultdict[str, int] = defaultdict(int)

        ranked_indexes = np.argsort([item.hybrid for item in similarity_scores])[::-1]
        for idx in ranked_indexes:
            source = candidate_values[idx]
            source_type = self._map_source_type(source.get("type"))
            score_parts = similarity_scores[idx]
            relevance = _safe_float(score_parts.hybrid)
            diversity = 1.0 if seen_types[source_type] == 0 else 0.6
            ecr_boost = self._source_ecr_boost(source)
            fairness_bonus = self._source_fairness_bonus(source)
            quality_score = self._source_quality_score(source)
            if quality_score < 0.18 and relevance < 0.2:
                continue
            final_score = self._combine_source_scores(relevance, diversity, ecr_boost, fairness_bonus, quality_score)
            seen_types[source_type] += 1
            recommendations.append(
                OpportunityRecommendation(
                    opportunity_id=source["id"].split("/")[-1],
                    opportunity_type=source_type,
                    title=source.get("display_name", "Unknown venue"),
                    score=final_score,
                    region=source.get("country_code") or "Global",
                    deadline=None,
                    description=self._build_source_description(source),
                    homepage_url=source.get("homepage_url"),
                    explanation=RecommendationExplanation(
                        relevance=relevance,
                        diversity=diversity,
                        ecr_boost=ecr_boost,
                        final_score=final_score,
                        reasons=[
                            *self._build_source_reasons(source, relevance, ecr_boost),
                            self._hybrid_reason(score_parts),
                            f"Fairness/open-access bonus: {fairness_bonus:.2f}",
                            f"Venue quality filter score: {quality_score:.2f}",
                        ],
                        model_scores=self._model_scores(score_parts),
                    ),
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _recommend_scientific_works(
        self,
        researcher_text: str,
        embedding_model: EmbeddingModelName,
        top_k: int,
        exclude_work_ids: list[str] | None = None,
    ) -> tuple[list[OpportunityRecommendation], str]:
        model_name = embedding_model if embedding_model in {"hashing", "scibert", "specter2"} else "hashing"
        label = self._embedding_model_label(model_name)
        try:
            model = self._get_embedding_model(model_name)
            query_text = researcher_text.strip() or "scientific profile recommendation"
            query_embedding = model.encode([query_text])[0].tolist()
            rows = self.research_db.search_similar_works(
                query_embedding,
                model_name=model_name,
                limit=max(top_k * 2, 8),
                exclude_work_ids=exclude_work_ids,
            )
        except Exception as exc:
            return [], f"{label} selected, but pgvector search is unavailable: {exc}"

        recommendations: list[OpportunityRecommendation] = []
        for row in rows:
            topics = self._db_topics(row.get("topics"))
            similarity = _safe_float(float(row.get("similarity") or 0.0))
            citation_signal = _safe_float(min(int(row.get("cited_by_count", 0) or 0) / 50, 1.0))
            topic_signal = _safe_float(min(len(topics) / 5, 1.0))
            final_score = round(_safe_float(0.76 * similarity + 0.14 * citation_signal + 0.1 * topic_signal), 4)
            recommendations.append(
                OpportunityRecommendation(
                    opportunity_id=str(row.get("work_id") or ""),
                    opportunity_type="paper",
                    title=row.get("title") or "Untitled OpenAlex work",
                    score=final_score,
                    region=str(row.get("publication_year") or "OpenAlex"),
                    deadline=None,
                    description=(
                        f"Similar OpenAlex paper found by {label} embeddings in pgvector. "
                        f"Venue: {row.get('source_name') or 'unknown venue'}."
                    ),
                    homepage_url=row.get("landing_page_url"),
                    explanation=RecommendationExplanation(
                        relevance=similarity,
                        diversity=topic_signal,
                        ecr_boost=citation_signal,
                        final_score=final_score,
                        reasons=[
                            f"{label} vector similarity: {similarity:.3f}",
                            f"Stored in PostgreSQL/pgvector with model_name='{model_name}'",
                            f"Topics: {', '.join(topics[:4]) if topics else 'not available'}",
                            f"Citation signal from OpenAlex: {citation_signal:.2f}",
                        ],
                        model_scores=[
                            MLModelScore(
                                name=f"{label} pgvector",
                                score=similarity,
                                role=self._embedding_model_role(model_name),
                            ),
                            MLModelScore(
                                name="Citation signal",
                                score=citation_signal,
                                role="impact/context signal",
                            ),
                            MLModelScore(
                                name="Topic coverage",
                                score=topic_signal,
                                role="metadata signal",
                            ),
                        ],
                    ),
                )
            )
            if len(recommendations) >= top_k:
                break

        if not recommendations:
            return [], f"{label} selected, but no pgvector work embeddings matched this profile."
        return recommendations, f"{label} active: pgvector returned {len(recommendations)} similar OpenAlex works."

    def _get_embedding_model(self, model_name: str) -> ScientificEmbeddingModel:
        if model_name not in self._embedding_models:
            self._embedding_models[model_name] = ScientificEmbeddingModel(model_name)
        return self._embedding_models[model_name]

    def _embedding_model_label(self, model_name: str) -> str:
        labels = {
            "hashing": "Hashing baseline",
            "scibert": "SciBERT",
            "specter2": "SPECTER2",
        }
        return labels.get(model_name, model_name)

    def _embedding_model_role(self, model_name: str) -> str:
        roles = {
            "hashing": "local embedding baseline",
            "scibert": "scientific BERT transformer",
            "specter2": "scientific paper embedding transformer",
        }
        return roles.get(model_name, "scientific embedding model")

    def _db_topics(self, value) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item]
            except Exception:
                return [cleaned]
        return []

    def _recommend_collaborators_from_queries(
        self,
        queries: list[str],
        researcher_text: str,
        top_k: int,
        current_author_id: str | None,
        current_author: dict | None = None,
        collaborator_scope: CollaboratorScope = "global",
        reference_affiliation: str | None = None,
        reference_country: str | None = None,
        embedding_model: EmbeddingModelName = "specter2",
        current_works: list[dict] | None = None,
    ) -> list[CollaboratorRecommendation]:
        self._graphsage_ranker.prepare()
        self._graphsage_status = self._graphsage_ranker.status
        candidate_map: dict[str, dict] = {}
        evidence_map: defaultdict[str, list[dict]] = defaultdict(list)
        target_affiliation = reference_affiliation or (_extract_affiliation(current_author) if current_author else None)
        target_country = reference_country or (_extract_country(current_author) if current_author else None)
        same_affiliation_ids = (
            self._resolve_institution_ids(target_affiliation)
            if collaborator_scope == "same_affiliation"
            else []
        )
        extra_filters = self._build_collaborator_scope_filters(
            collaborator_scope,
            reference_affiliation=target_affiliation,
            reference_country=target_country,
            institution_ids=same_affiliation_ids,
        )

        for query in queries:
            for work in self.client.search_works(query, per_page=16, extra_filters=extra_filters):
                for candidate in self._extract_author_candidates_from_work(work):
                    candidate_id = candidate["id"].split("/")[-1]
                    if current_author_id and candidate_id == current_author_id:
                        continue
                    candidate_map[candidate_id] = self._merge_candidate(candidate_map.get(candidate_id), candidate)
                    evidence_map[candidate_id].append(work)

        if collaborator_scope == "same_affiliation":
            self._add_raw_affiliation_work_candidates(
                candidate_map,
                evidence_map,
                queries=queries,
                current_author_id=current_author_id,
                target_affiliation=target_affiliation,
                target_country=target_country,
            )
            self._add_same_affiliation_author_candidates(
                candidate_map,
                queries=queries,
                current_author=current_author,
                current_author_id=current_author_id,
                institution_ids=same_affiliation_ids,
            )

        if not candidate_map:
            return []

        candidate_values = [
            candidate
            for candidate in candidate_map.values()
            if self._candidate_matches_scope(
                candidate,
                collaborator_scope,
                current_author=current_author,
                reference_affiliation=reference_affiliation,
                reference_country=reference_country,
            )
        ]
        if not candidate_values:
            return []

        texts = [
            self._build_collaborator_candidate_text(
                item,
                self._dedupe_works(evidence_map.get(item["id"].split("/")[-1], [])),
            )
            for item in candidate_values
        ]
        similarity_scores = self.scorer.score(researcher_text, texts)
        transformer_scores = self._collaborator_transformer_scores(
            researcher_text,
            texts,
            embedding_model,
        )

        recommendations: list[CollaboratorRecommendation] = []

        ranked_indexes = np.argsort([item.hybrid for item in similarity_scores])[::-1]
        for idx in ranked_indexes:
            candidate = candidate_values[idx]
            candidate_id = candidate["id"].split("/")[-1]
            evidence_works = self._dedupe_works(evidence_map.get(candidate_id, []))
            score_parts = similarity_scores[idx]
            relevance = _safe_float(score_parts.hybrid)
            has_work_evidence = bool(evidence_works)
            graph_score = self._graph_collaboration_score(current_author, candidate, evidence_works)
            if relevance < MIN_COLLABORATOR_RELEVANCE and (
                has_work_evidence or collaborator_scope != "same_affiliation"
            ):
                continue
            if current_author and has_work_evidence and graph_score < 0.25:
                continue
            diversity = self._collaborator_diversity(current_author, candidate) if current_author else 0.8
            ecr_boost = self._collaborator_ecr_boost(candidate)
            fairness_bonus = self._collaborator_fairness_bonus(candidate)
            context_works = [*(current_works or []), *evidence_works]
            graphsage_score, graphsage_mode = self._graphsage_ranker.score_pair(
                current_author,
                candidate,
                context_works,
            )
            self._graphsage_status = self._graphsage_ranker.status
            transformer_score = transformer_scores[idx]
            final_score = self._combine_collaborator_scores(
                relevance,
                graph_score,
                diversity,
                ecr_boost,
                fairness_bonus,
                graphsage_score,
                transformer_score,
            )
            recommendations.append(
                CollaboratorRecommendation(
                    researcher_id=candidate_id,
                    name=candidate.get("display_name", "Unknown author"),
                    affiliation=_extract_affiliation(candidate),
                    country_code=_extract_country(candidate),
                    score=final_score,
                    scope_match=self._scope_match_label(
                        candidate,
                        collaborator_scope,
                        current_author=current_author,
                        reference_affiliation=reference_affiliation,
                        reference_country=reference_country,
                    ),
                    evidence_works=[
                        self._map_work_summary(work)
                        for work in evidence_works[:3]
                    ],
                    explanation=RecommendationExplanation(
                        relevance=relevance,
                        diversity=diversity,
                        ecr_boost=ecr_boost,
                        final_score=final_score,
                        reasons=[
                            *self._build_collaborator_reasons(
                                current_author,
                                candidate,
                                diversity,
                                ecr_boost,
                                evidence_works,
                            ),
                            self._hybrid_reason(score_parts),
                            f"Graph collaboration evidence score: {graph_score:.2f}",
                            self._graphsage_reason(graphsage_score, graphsage_mode),
                            self._collaborator_transformer_reason(transformer_score, embedding_model),
                            f"Fairness-aware ECR exposure bonus: {fairness_bonus:.2f}",
                        ],
                        model_scores=self._model_scores(
                            score_parts,
                            graphsage_score,
                            transformer_score,
                            embedding_model,
                        ),
                    ),
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _extract_author_candidates_from_work(
        self,
        work: dict,
        target_affiliation: str | None = None,
        target_country: str | None = None,
        strict_affiliation: bool = False,
    ) -> list[dict]:
        candidates = []
        work_topics = work.get("topics", [])
        for authorship in work.get("authorships", [])[:8]:
            author = authorship.get("author") or {}
            author_id = author.get("id")
            if not author_id:
                continue
            raw_affiliations = authorship.get("raw_affiliation_strings") or []
            raw_affiliation_match = self._raw_affiliation_matches(raw_affiliations, target_affiliation)
            institution_match = _affiliations_match(_extract_affiliation_from_authorship(authorship), target_affiliation)
            if strict_affiliation and not (raw_affiliation_match or institution_match):
                continue
            affiliation = target_affiliation if raw_affiliation_match and target_affiliation else _extract_affiliation_from_authorship(authorship)
            country = (
                _normalize_country(target_country)
                if raw_affiliation_match and target_country
                else _extract_country_from_authorship(authorship)
            )
            candidates.append(
                {
                    "id": author_id,
                    "display_name": author.get("display_name", "Unknown author"),
                    "works_count": int(author.get("works_count", 0) or 0),
                    "cited_by_count": int(author.get("cited_by_count", 0) or 0),
                    "last_known_institutions": [
                        {
                            "display_name": affiliation,
                            "country_code": country,
                        }
                    ],
                    "affiliation_source": "raw_affiliation" if raw_affiliation_match else "openalex_institution",
                    "topics": work_topics,
                }
            )
        return candidates

    def _raw_affiliation_matches(self, raw_affiliations: list[str], target_affiliation: str | None) -> bool:
        if not target_affiliation:
            return False
        normalized_target = _normalize_affiliation(target_affiliation)
        return any(
            normalized_target in _normalize_affiliation(raw_value)
            or _affiliations_match(raw_value, target_affiliation)
            for raw_value in raw_affiliations
        )

    def _add_raw_affiliation_work_candidates(
        self,
        candidate_map: dict[str, dict],
        evidence_map: defaultdict[str, list[dict]],
        queries: list[str],
        current_author_id: str | None,
        target_affiliation: str | None,
        target_country: str | None,
    ) -> None:
        if not target_affiliation:
            return
        raw_filter = f"raw_affiliation_strings.search:{target_affiliation}"
        for query in queries:
            for work in self.client.search_works(query, per_page=16, extra_filters=[raw_filter]):
                for candidate in self._extract_author_candidates_from_work(
                    work,
                    target_affiliation=target_affiliation,
                    target_country=target_country,
                    strict_affiliation=True,
                ):
                    candidate_id = candidate["id"].split("/")[-1]
                    if current_author_id and candidate_id == current_author_id:
                        continue
                    candidate_map[candidate_id] = self._merge_candidate(candidate_map.get(candidate_id), candidate)
                    evidence_map[candidate_id].append(work)

    def _add_same_affiliation_author_candidates(
        self,
        candidate_map: dict[str, dict],
        queries: list[str],
        current_author: dict | None,
        current_author_id: str | None,
        institution_ids: list[str],
    ) -> None:
        if not institution_ids:
            return

        institution_filter = f"last_known_institutions.id:{'|'.join(institution_ids)}"
        topic_ids = _topic_ids((current_author or {}).get("topics", []), limit=4)
        filter_sets = [[institution_filter]]
        if topic_ids:
            filter_sets.insert(0, [institution_filter, f"topics.id:{'|'.join(topic_ids)}"])

        for filters in filter_sets:
            added_before = len(candidate_map)
            for author in self.client.search_authors(
                None,
                per_page=24,
                extra_filters=filters,
                sort="works_count:desc",
            ):
                candidate_id = author.get("id", "").split("/")[-1]
                if not candidate_id or (current_author_id and candidate_id == current_author_id):
                    continue
                candidate_map[candidate_id] = self._merge_candidate(candidate_map.get(candidate_id), author)
            if len(candidate_map) > added_before:
                break

    def _merge_candidate(self, existing: dict | None, candidate: dict) -> dict:
        if not existing:
            return candidate
        merged = {**existing}
        for key in ["display_name", "works_count", "cited_by_count"]:
            if not merged.get(key) and candidate.get(key):
                merged[key] = candidate[key]
        existing_topics = merged.get("topics") or []
        seen_topics = {item.get("display_name") for item in existing_topics}
        merged["topics"] = [
            *existing_topics,
            *[
                item
                for item in candidate.get("topics", [])
                if item.get("display_name") and item.get("display_name") not in seen_topics
            ],
        ][:8]
        if candidate.get("affiliation_source") == "raw_affiliation":
            merged["last_known_institutions"] = candidate.get("last_known_institutions", [])
            merged["affiliation_source"] = "raw_affiliation"
        if _extract_affiliation(merged) == "Unknown affiliation" and _extract_affiliation(candidate) != "Unknown affiliation":
            merged["last_known_institutions"] = candidate.get("last_known_institutions", [])
        return merged

    def _dedupe_works(self, works: list[dict]) -> list[dict]:
        deduped = []
        seen = set()
        for work in works:
            work_id = work.get("id")
            if not work_id or work_id in seen:
                continue
            seen.add(work_id)
            deduped.append(work)
        return deduped

    def _build_collaborator_candidate_text(self, candidate: dict, evidence_works: list[dict]) -> str:
        parts = [
            candidate.get("display_name", ""),
            _extract_affiliation(candidate),
            " ".join(_topic_names(candidate.get("topics", []), limit=8)),
        ]
        for work in evidence_works[:5]:
            parts.append(work.get("title", ""))
            parts.append(" ".join(_topic_names(work.get("topics", []), limit=6)))
            source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            if source:
                parts.append(source)
        return " ".join(part for part in parts if part).strip()

    def _build_registered_author_text(
        self,
        profile: RegisteredResearcherProfile,
        author: dict,
        works: list[dict],
    ) -> str:
        text_parts = [
            profile.full_name,
            profile.affiliation,
            " ".join(_topic_names(author.get("topics", []), limit=8)),
        ]
        for work in works[:8]:
            text_parts.append(work.get("title", ""))
            text_parts.append(" ".join(_topic_names(work.get("topics", []), limit=4)))
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            text_parts.append(source.get("display_name", ""))
        return " ".join(part for part in text_parts if part).strip()

    def _merge_queries(self, queries: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for query in queries:
            cleaned = query.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                deduped.append(cleaned)
        return deduped[:7]

    def _build_collaborator_scope_filters(
        self,
        collaborator_scope: CollaboratorScope,
        reference_affiliation: str | None,
        reference_country: str | None,
        institution_ids: list[str] | None = None,
    ) -> list[str]:
        if collaborator_scope == "same_country":
            country = _normalize_country(reference_country)
            return [f"institutions.country_code:{country}"] if country else []
        if collaborator_scope == "same_affiliation":
            resolved_ids = institution_ids if institution_ids is not None else self._resolve_institution_ids(reference_affiliation)
            if resolved_ids:
                joined_ids = "|".join(resolved_ids)
                return [f"institutions.id:{joined_ids}"]
        return []

    def _resolve_institution_ids(self, affiliation: str | None) -> list[str]:
        if not affiliation:
            return []
        try:
            institutions = self.client.search_institutions(affiliation, per_page=5)
        except Exception:
            return []

        matched_ids: list[str] = []
        for institution in institutions:
            display_name = institution.get("display_name", "")
            if not _affiliations_match(display_name, affiliation):
                continue
            institution_id = institution.get("id")
            if institution_id:
                matched_ids.append(institution_id)
        return matched_ids[:3]

    def _scope_match_label(
        self,
        candidate: dict,
        collaborator_scope: CollaboratorScope,
        current_author: dict | None,
        reference_affiliation: str | None,
        reference_country: str | None,
    ) -> str:
        if collaborator_scope == "same_affiliation":
            target = reference_affiliation or (_extract_affiliation(current_author) if current_author else "")
            return f"same affiliation: {target}" if target else "same affiliation"
        if collaborator_scope == "same_country":
            target = _normalize_country(reference_country or (_extract_country(current_author) if current_author else ""))
            return f"same country: {target}" if target else "same country"
        candidate_country = _normalize_country(_extract_country(candidate))
        return f"global candidate{f' ({candidate_country})' if candidate_country else ''}"

    def _candidate_matches_scope(
        self,
        candidate: dict,
        collaborator_scope: CollaboratorScope,
        current_author: dict | None,
        reference_affiliation: str | None,
        reference_country: str | None,
    ) -> bool:
        if collaborator_scope == "global":
            return True

        target_affiliation = reference_affiliation or (_extract_affiliation(current_author) if current_author else "")
        target_country = reference_country or (_extract_country(current_author) if current_author else "")

        if collaborator_scope == "same_affiliation":
            return _affiliations_match(_extract_affiliation(candidate), target_affiliation)

        if collaborator_scope == "same_country":
            candidate_country = _normalize_country(_extract_country(candidate))
            normalized_target = _normalize_country(target_country)
            return bool(candidate_country and normalized_target and candidate_country == normalized_target)

        return True

    def _build_topic_queries(self, author: dict, works: list[dict]) -> list[str]:
        queries: list[str] = []
        queries.extend(_topic_names(author.get("topics", []), limit=3))
        for work in works[:6]:
            queries.extend(_topic_names(work.get("topics", []), limit=2))
        deduped = []
        seen = set()
        for item in queries:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped[:4] or [author.get("display_name", "research")]

    def _build_source_text(self, source: dict) -> str:
        return " ".join(
            [
                source.get("display_name") or "",
                source.get("type") or "",
                " ".join(_topic_names(source.get("topics", []), limit=8)),
                source.get("host_organization_name") or "",
                source.get("country_code") or "",
            ]
        )

    def _build_source_description(self, source: dict) -> str:
        topics = ", ".join(_topic_names(source.get("topics", []), limit=4))
        summary = source.get("summary_stats") or {}
        return (
            f"Real OpenAlex source. Topics: {topics or 'not available'}. "
            f"h-index: {int(summary.get('h_index', 0))}, "
            f"works: {int(source.get('works_count', 0))}."
        )

    def _build_source_reasons(self, source: dict, relevance: float, ecr_boost: float) -> list[str]:
        reasons = []
        topics = _topic_names(source.get("topics", []), limit=3)
        if topics:
            reasons.append(f"Topic alignment with source areas: {', '.join(topics)}")
        if relevance >= 0.25:
            reasons.append("Semantic match between researcher profile and venue scope")
        if ecr_boost >= 0.6:
            reasons.append("Open or lower-barrier venue characteristics can help early visibility")
        return reasons or ["Real venue recommendation based on topic overlap"]

    def _build_collaborator_reasons(
        self,
        author: dict | None,
        candidate: dict,
        diversity: float,
        ecr_boost: float,
        evidence_works: list[dict] | None = None,
    ) -> list[str]:
        reasons = []
        author_topics = _topic_names(author.get("topics", []), 6) if author else []
        shared = sorted(set(author_topics) & set(_topic_names(candidate.get("topics", []), 6)))
        if shared:
            reasons.append(f"Shared topics: {', '.join(shared[:3])}")
        if evidence_works:
            title = evidence_works[0].get("title") or "recent OpenAlex work"
            reasons.append(f"Found through real OpenAlex work: {title}")
        if _normalize_country(_extract_country(candidate)):
            reasons.append(f"Candidate country: {_normalize_country(_extract_country(candidate))}")
        if diversity >= 0.55:
            reasons.append("Institutional or geographic diversity can expand visibility")
        if ecr_boost >= 0.6:
            reasons.append("Candidate has a career stage that may fit collaboration growth")
        return reasons or ["Real collaborator candidate from OpenAlex topic search"]

    def _build_analytics(self, author: dict, works: list[dict]) -> DashboardAnalytics:
        works_count = int(author.get("works_count", 0))
        cited_by_count = int(author.get("cited_by_count", 0))
        h_index = int((author.get("summary_stats") or {}).get("h_index", 0))
        first_years = [work.get("publication_year") for work in works if work.get("publication_year")]
        earliest_recent_year = min(first_years) if first_years else CURRENT_YEAR
        ecr_status = CURRENT_YEAR - int(earliest_recent_year) <= 5
        profile_strength = _safe_float((works_count / 25) * 0.45 + (h_index / 25) * 0.3 + (cited_by_count / 500) * 0.25)
        cold_start_risk = _safe_float(1.0 - ((works_count / 20) * 0.7 + (cited_by_count / 200) * 0.3))
        collaboration_readiness = _safe_float(
            0.35 + min(len(_topic_names(author.get("topics", []), 8)), 6) / 10 + (0.1 if ecr_status else 0.0)
        )
        return DashboardAnalytics(
            profile_strength=round(profile_strength, 4),
            cold_start_risk=round(cold_start_risk, 4),
            ecr_status=ecr_status,
            publication_count=works_count,
            collaboration_readiness=round(collaboration_readiness, 4),
        )

    def _map_source_type(self, raw_type: str | None) -> str:
        if raw_type == "conference":
            return "conference"
        if raw_type == "repository":
            return "repository"
        return "journal"

    def _source_ecr_boost(self, source: dict) -> float:
        is_open = bool(source.get("is_oa")) or bool(source.get("is_in_doaj"))
        lower_volume = int(source.get("works_count", 0)) < 5000
        return _safe_float((0.6 if is_open else 0.2) + (0.2 if lower_volume else 0.0))

    def _source_fairness_bonus(self, source: dict) -> float:
        works_count = int(source.get("works_count", 0))
        summary = source.get("summary_stats") or {}
        h_index = int(summary.get("h_index", 0))
        accessible = 1.0 if bool(source.get("is_oa")) or bool(source.get("is_in_doaj")) else 0.35
        scale_bonus = 1.0 - min(works_count / 50_000, 1.0)
        prestige_balance = 1.0 - min(h_index / 250, 1.0)
        return _safe_float(0.45 * accessible + 0.35 * scale_bonus + 0.2 * prestige_balance)

    def _source_quality_score(self, source: dict) -> float:
        summary = source.get("summary_stats") or {}
        h_index = int(summary.get("h_index", 0))
        works_count = int(source.get("works_count", 0))
        topic_signal = min(len(_topic_names(source.get("topics", []), limit=8)) / 5, 1.0)
        impact_signal = min(h_index / 100, 1.0)
        volume_signal = min(works_count / 1500, 1.0)
        access_signal = 1.0 if bool(source.get("is_oa")) or bool(source.get("is_in_doaj")) else 0.45
        return _safe_float(0.4 * impact_signal + 0.25 * volume_signal + 0.2 * topic_signal + 0.15 * access_signal)

    def _collaborator_diversity(self, author: dict, candidate: dict) -> float:
        if not author:
            return 0.8
        same_institution = _affiliations_match(_extract_affiliation(author), _extract_affiliation(candidate))
        same_country = _normalize_country(_extract_country(author)) == _normalize_country(_extract_country(candidate))
        return _safe_float(1.0 - (0.45 if same_institution else 0.0) - (0.2 if same_country else 0.0))

    def _collaborator_ecr_boost(self, candidate: dict) -> float:
        works_count = int(candidate.get("works_count", 0))
        return _safe_float(0.85 if works_count <= 25 else 0.35)

    def _collaborator_fairness_bonus(self, candidate: dict) -> float:
        works_count = int(candidate.get("works_count", 0))
        cited_by_count = int(candidate.get("cited_by_count", 0))
        career_stage_signal = 1.0 - min(works_count / 80, 1.0)
        citation_exposure_signal = 1.0 - min(cited_by_count / 500, 1.0)
        return _safe_float(0.45 * career_stage_signal + 0.55 * citation_exposure_signal)

    def _graph_collaboration_score(
        self,
        author: dict | None,
        candidate: dict,
        evidence_works: list[dict],
    ) -> float:
        author_topics = set(_topic_names((author or {}).get("topics", []), limit=8))
        candidate_topics = set(_topic_names(candidate.get("topics", []), limit=8))
        evidence_topics: set[str] = set()
        for work in evidence_works:
            evidence_topics.update(_topic_names(work.get("topics", []), limit=6))

        shared_with_author = author_topics & (candidate_topics | evidence_topics)
        candidate_evidence_overlap = candidate_topics & evidence_topics
        if author_topics:
            topic_signal = min(len(shared_with_author) / 4, 1.0)
        else:
            topic_signal = min(len(candidate_evidence_overlap) / 4, 1.0)
        evidence_signal = min(len(evidence_works) / 3, 1.0)

        if author:
            same_institution = _affiliations_match(_extract_affiliation(author), _extract_affiliation(candidate))
            same_country = _normalize_country(_extract_country(author)) == _normalize_country(_extract_country(candidate))
            proximity_signal = 1.0 if same_institution else 0.7 if same_country else 0.45
        else:
            proximity_signal = 0.55

        return _safe_float(0.45 * topic_signal + 0.35 * evidence_signal + 0.2 * proximity_signal)

    def _collaborator_transformer_scores(
        self,
        researcher_text: str,
        candidate_texts: list[str],
        embedding_model: EmbeddingModelName,
    ) -> list[float | None]:
        if embedding_model == "hashing":
            self._collaborator_semantic_status = "Hashing semantic score is included in the hybrid collaborator ranker."
            return [None] * len(candidate_texts)
        label = self._embedding_model_label(embedding_model)
        try:
            model = self._get_embedding_model(embedding_model)
            query = model.encode([researcher_text])[0]
            candidates = model.encode(candidate_texts)
            query = query / max(float(np.linalg.norm(query)), 1e-8)
            candidates = candidates / np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-8)
            scores = np.clip((candidates @ query + 1.0) / 2.0, 0.0, 1.0)
            self._collaborator_semantic_status = (
                f"{label} live collaborator semantic ranking active for {len(candidate_texts)} candidates."
            )
            return [round(float(score), 4) for score in scores]
        except Exception as exc:
            self._collaborator_semantic_status = f"{label} collaborator ranking unavailable: {exc}"
            return [None] * len(candidate_texts)

    def _graphsage_reason(self, graphsage_score: float | None, mode: str) -> str:
        if graphsage_score is None:
            return "GraphSAGE could not infer this author pair; legacy graph evidence used."
        if mode == "inductive":
            return f"GraphSAGE inductive similarity for a new live author pair: {graphsage_score:.2f}."
        return f"GraphSAGE author-embedding similarity: {graphsage_score:.2f}."

    def _collaborator_model_status(self) -> str:
        return f"{self._graphsage_status} {self._collaborator_semantic_status}"

    def _collaborator_transformer_reason(
        self,
        transformer_score: float | None,
        embedding_model: EmbeddingModelName,
    ) -> str:
        if transformer_score is None:
            if embedding_model == "hashing":
                return "Transformer collaborator score not selected; semantic hashing remains in the hybrid ranker."
            return f"{self._embedding_model_label(embedding_model)} collaborator score unavailable; fallback signals used."
        return f"{self._embedding_model_label(embedding_model)} collaborator semantic similarity: {transformer_score:.2f}."

    def _combine_source_scores(
        self,
        relevance: float,
        diversity: float,
        ecr_boost: float,
        fairness_bonus: float,
        quality_score: float,
    ) -> float:
        final_score = (
            0.42 * relevance
            + 0.16 * diversity
            + 0.14 * ecr_boost
            + 0.1 * fairness_bonus
            + 0.18 * quality_score
        )
        return round(_safe_float(final_score), 4)

    def _combine_collaborator_scores(
        self,
        relevance: float,
        graph_score: float,
        diversity: float,
        ecr_boost: float,
        fairness_bonus: float,
        graphsage_score: float | None = None,
        transformer_score: float | None = None,
    ) -> float:
        if graphsage_score is not None and transformer_score is not None:
            final_score = (
                0.27 * relevance
                + 0.16 * graph_score
                + 0.18 * graphsage_score
                + 0.23 * transformer_score
                + 0.08 * diversity
                + 0.04 * ecr_boost
                + 0.04 * fairness_bonus
            )
        elif graphsage_score is not None:
            final_score = (
                0.36 * relevance
                + 0.18 * graph_score
                + 0.22 * graphsage_score
                + 0.12 * diversity
                + 0.06 * ecr_boost
                + 0.06 * fairness_bonus
            )
        elif transformer_score is not None:
            final_score = (
                0.36 * relevance
                + 0.22 * graph_score
                + 0.22 * transformer_score
                + 0.12 * diversity
                + 0.04 * ecr_boost
                + 0.04 * fairness_bonus
            )
        else:
            final_score = (
                0.42 * relevance
                + 0.24 * graph_score
                + 0.14 * diversity
                + 0.1 * ecr_boost
                + 0.1 * fairness_bonus
            )
        return round(_safe_float(final_score), 4)

    def _combine_scores(self, relevance: float, diversity: float, ecr_boost: float) -> float:
        final_score = (
            settings.relevance_weight * relevance
            + settings.diversity_weight * diversity
            + settings.ecr_weight * ecr_boost
        )
        return round(_safe_float(final_score), 4)

    def _hybrid_reason(self, score_parts: HybridSimilarityScore) -> str:
        return (
            "Model comparison: "
            f"TF-IDF {score_parts.lexical:.2f}, "
            f"BM25 {score_parts.bm25:.2f}, "
            f"LSA/SVD {score_parts.lsa:.2f}, "
            f"semantic hashing {score_parts.semantic:.2f}, "
            f"hybrid ensemble {score_parts.hybrid:.2f}."
        )

    def _model_scores(
        self,
        score_parts: HybridSimilarityScore,
        graphsage_score: float | None = None,
        transformer_score: float | None = None,
        embedding_model: EmbeddingModelName = "hashing",
    ) -> list[MLModelScore]:
        scores = [
            MLModelScore(name=item.name, score=item.score, role=item.role)
            for item in score_parts.model_scores
        ]
        if graphsage_score is not None:
            scores.append(
                MLModelScore(
                    name="GraphSAGE",
                    score=graphsage_score,
                    role="co-author graph neural embedding similarity",
                )
            )
        if transformer_score is not None:
            scores.append(
                MLModelScore(
                    name=self._embedding_model_label(embedding_model),
                    score=transformer_score,
                    role="live scientific-text similarity for collaborator ranking",
                )
            )
        return scores

    def _map_registered_profile(self, profile: RegisteredResearcherProfile) -> ResearcherProfile:
        derived = profile.derived_metrics
        return ResearcherProfile(
            researcher_id=profile.profile_id,
            name=profile.full_name,
            works_count=int(derived.get("works_count", 0)),
            cited_by_count=int(derived.get("citation_count", 0)),
            h_index=int(derived.get("h_index", 0)),
            affiliation=profile.affiliation,
            country_code=None,
            homepage_url=profile.raw_profile.personal_website_url,
            orcid=profile.raw_profile.orcid_id,
            topics=[TopicSummary(name=item.name, score=item.score) for item in profile.domain_candidates],
        )

    def _build_registered_profile_text(self, profile: RegisteredResearcherProfile) -> str:
        raw = profile.raw_profile
        parts = [
            profile.full_name,
            profile.affiliation,
            profile.academic_field,
            raw.position_title,
            raw.academic_level,
            " ".join(profile.research_keywords),
            " ".join(profile.thematic_clusters),
            " ".join(raw.methods),
            " ".join(raw.collaboration_goals),
            " ".join(raw.publication_goals),
            profile.interest_embedding_text,
        ]
        return " ".join(part for part in parts if part).strip()

    def _build_registered_queries(self, profile: RegisteredResearcherProfile) -> list[str]:
        raw = profile.raw_profile
        precise_terms = [
            *profile.research_keywords[:3],
            *profile.thematic_clusters[:2],
            *raw.methods[:2],
        ]
        queries = [
            " ".join(precise_terms[:4]),
            " ".join([*profile.research_keywords[:2], *profile.thematic_clusters[:1]]),
            *profile.research_keywords[:3],
            *profile.thematic_clusters[:2],
            *raw.methods[:2],
            *raw.collaboration_goals[:2],
            *raw.publication_goals[:1],
        ]
        if not any(item.strip() for item in queries):
            queries.append(profile.academic_field)
        deduped = []
        seen = set()
        for item in queries:
            cleaned = item.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                deduped.append(cleaned)
        return deduped[:5] or [profile.academic_field]

    def _build_registered_analytics(self, profile: RegisteredResearcherProfile) -> DashboardAnalytics:
        derived = profile.derived_metrics
        works_count = int(derived.get("works_count", 0))
        cited_by_count = int(derived.get("citation_count", 0))
        h_index = int(derived.get("h_index", 0))
        ecr_status = works_count <= 25
        profile_strength = _safe_float((works_count / 25) * 0.45 + (h_index / 25) * 0.3 + (cited_by_count / 500) * 0.25)
        cold_start_risk = _safe_float(1.0 - ((works_count / 20) * 0.7 + (cited_by_count / 200) * 0.3))
        collaboration_readiness = _safe_float(
            0.35 + min(len(profile.domain_candidates), 6) / 10 + (0.1 if ecr_status else 0.0)
        )
        return DashboardAnalytics(
            profile_strength=round(profile_strength, 4),
            cold_start_risk=round(cold_start_risk, 4),
            ecr_status=ecr_status,
            publication_count=works_count,
            collaboration_readiness=round(collaboration_readiness, 4),
        )

    def _build_connected_profiles(self, profile: RegisteredResearcherProfile) -> list[ConnectedProfile]:
        raw = profile.raw_profile
        candidates = [
            ("ORCID", raw.orcid_id, "Connected academic identifier used for OpenAlex enrichment."),
            ("Google Scholar", raw.google_scholar_url, "Reference link saved for future enrichment."),
            ("ResearchGate", raw.researchgate_url, "Reference link saved for future enrichment."),
            ("Personal website", raw.personal_website_url, "Optional external profile or lab page."),
        ]
        connected_profiles: list[ConnectedProfile] = []
        for platform, url, note in candidates:
            if url:
                connected_profiles.append(
                    ConnectedProfile(platform=platform, url=url, status="connected", note=note)
                )
        return connected_profiles

    def _build_profile_basis(self, profile: RegisteredResearcherProfile, mode: str) -> str:
        if mode == "orcid_openalex":
            return "Recommendations are based on your ORCID-linked OpenAlex publications, topics, and venue history."
        return (
            "Recommendations are based on your registration profile: academic field, keywords, methods, bio, and goals. "
            "This is a cold-start mode for users without connected publication history."
        )

    def _build_recommendation_steps(self, mode: str) -> list[str]:
        steps = [
            "The system builds a profile text from your academic field, keywords, methods, and goals.",
            "It searches real OpenAlex sources and authors related to that profile.",
            "It ranks venues and collaborators with TF-IDF, BM25, LSA/SVD, graph evidence, diversity, and fairness-aware reranking.",
            "It also runs the selected embedding model against pgvector to recommend similar OpenAlex papers.",
        ]
        if mode == "orcid_openalex":
            steps.insert(1, "Your ORCID-linked publications and topics are added to strengthen the semantic profile.")
        else:
            steps.insert(1, "Because there is no verified publication history, the system uses cold-start reasoning.")
        return steps

    def _build_promotion_tracker(
        self,
        profile: RegisteredResearcherProfile,
        dashboard: DashboardResponse,
    ) -> PromotionTracker:
        analytics = dashboard.analytics
        completeness = profile.profile_completeness
        promotion_score = _safe_float(
            0.3 * analytics.profile_strength
            + 0.25 * (1.0 - analytics.cold_start_risk)
            + 0.25 * analytics.collaboration_readiness
            + 0.2 * completeness
        )
        visibility_gap = _safe_float(1.0 - promotion_score)
        citation_growth_potential = _safe_float(
            0.5 * (1.0 - analytics.profile_strength) + 0.5 * min(len(dashboard.opportunities) / 5, 1.0)
        )
        collaboration_growth_potential = _safe_float(
            0.55 * analytics.collaboration_readiness + 0.45 * min(len(dashboard.collaborators) / 5, 1.0)
        )
        gap_dimensions = self._build_visibility_gap_dimensions(profile, dashboard)
        roadmap = self._build_promotion_roadmap(profile, dashboard)
        actions = self._build_promotion_actions(profile, dashboard, visibility_gap)
        return PromotionTracker(
            promotion_score=round(promotion_score, 4),
            visibility_gap=round(visibility_gap, 4),
            citation_growth_potential=round(citation_growth_potential, 4),
            collaboration_growth_potential=round(collaboration_growth_potential, 4),
            visibility_twin_summary=self._visibility_twin_summary(profile, gap_dimensions),
            gap_dimensions=gap_dimensions,
            roadmap=roadmap,
            ml_signals=self._build_ml_signals(dashboard),
            recommended_actions=actions,
            next_review_window="Recalculate after 30 days or after profile/publication updates.",
        )

    def _build_ml_signals(self, dashboard: DashboardResponse) -> list[MLSignal]:
        return [
            MLSignal(
                name="Model comparison layer",
                value=self.scorer.model_name,
                description="Compares TF-IDF, BM25, LSA/SVD, semantic hashing, and the final hybrid ensemble.",
            ),
            MLSignal(
                name="Transformer embeddings",
                value=f"{dashboard.embedding_model.upper()} active",
                description=dashboard.embedding_model_status,
            ),
            MLSignal(
                name="Graph collaboration layer",
                value=f"{len(dashboard.collaborators)} candidates",
                description="Uses OpenAlex works, authorship evidence, shared topics, country, and affiliation proximity.",
            ),
            MLSignal(
                name="Fairness-aware reranking",
                value="enabled",
                description="Adds exposure to relevant early-career or lower-citation candidates instead of only famous authors.",
            ),
            MLSignal(
                name="Promotion progress tracker",
                value="snapshot-ready",
                description="Stores profile snapshots so progress can be measured as score and metric deltas over time.",
            ),
        ]

    def _build_visibility_gap_dimensions(
        self,
        profile: RegisteredResearcherProfile,
        dashboard: DashboardResponse,
    ) -> list[VisibilityGapDimension]:
        connected_signal = min(len(self._build_connected_profiles(profile)) / 3, 1.0)
        keyword_signal = min(len(profile.research_keywords) / 6, 1.0)
        venue_signal = min(len(dashboard.opportunities) / 5, 1.0)
        collaborator_signal = min(len(dashboard.collaborators) / 5, 1.0)
        openalex_signal = 1.0 if "openalex_orcid" in profile.profile_data_sources else 0.25
        dimensions = [
            (
                "Verified identity",
                connected_signal,
                "ORCID, Google Scholar, ResearchGate or website links strengthen academic identity.",
            ),
            (
                "Semantic discoverability",
                keyword_signal,
                "Keywords and OpenAlex topics should clearly describe the profile's searchable niche.",
            ),
            (
                "Venue strategy",
                venue_signal,
                "The profile needs ranked target venues aligned with recent works and goals.",
            ),
            (
                "Collaboration network",
                collaborator_signal,
                "A young researcher needs topic-aligned candidates, local and global.",
            ),
            (
                "Verified publication graph",
                openalex_signal,
                "ORCID-linked OpenAlex works reduce cold-start uncertainty.",
            ),
        ]
        return [
            VisibilityGapDimension(
                name=name,
                current=round(current, 4),
                target=1.0,
                gap=round(_safe_float(1.0 - current), 4),
                explanation=explanation,
            )
            for name, current, explanation in dimensions
        ]

    def _build_promotion_roadmap(
        self,
        profile: RegisteredResearcherProfile,
        dashboard: DashboardResponse,
    ) -> list[PromotionRoadmapStep]:
        top_topic = dashboard.researcher.topics[0].name if dashboard.researcher.topics else profile.academic_field
        top_venue = dashboard.opportunities[0].title if dashboard.opportunities else "a topic-aligned venue"
        top_collaborator = dashboard.collaborators[0].name if dashboard.collaborators else "a topic-aligned collaborator"
        return [
            PromotionRoadmapStep(
                phase="Week 1",
                title="Normalize academic identity",
                objective="Connect identifiers and make profile metadata consistent across platforms.",
                metric="Profile completeness and verified identity gap",
            ),
            PromotionRoadmapStep(
                phase="Weeks 2-3",
                title=f"Own the niche: {top_topic}",
                objective="Update keywords, bio, and topics so search systems map the researcher to the right cluster.",
                metric="Semantic discoverability gap",
            ),
            PromotionRoadmapStep(
                phase="Month 1",
                title=f"Target venue: {top_venue}",
                objective="Prepare the next manuscript or preprint for the strongest semantic venue match.",
                metric="Venue strategy score and citation potential",
            ),
            PromotionRoadmapStep(
                phase="Month 1-2",
                title=f"Activate collaboration with {top_collaborator}",
                objective="Use real OpenAlex evidence to start a collaboration path instead of generic networking.",
                metric="Collaboration growth potential",
            ),
        ]

    def _visibility_twin_summary(
        self,
        profile: RegisteredResearcherProfile,
        gap_dimensions: list[VisibilityGapDimension],
    ) -> str:
        largest_gap = max(gap_dimensions, key=lambda item: item.gap, default=None)
        if not largest_gap:
            return "Visibility Twin could not identify a dominant gap yet."
        if largest_gap.gap <= 0.05:
            return (
                f"Visibility Twin compares {profile.full_name}'s current digital academic footprint "
                "with a target early-career visibility profile. No critical visibility gaps were detected; "
                "the next step is to maintain the profile and track progress over time."
            )
        return (
            f"Visibility Twin compares {profile.full_name}'s current digital academic footprint "
            f"with a target early-career visibility profile. The largest gap is "
            f"{largest_gap.name.lower()} ({largest_gap.gap})."
        )

    def _build_promotion_actions(
        self,
        profile: RegisteredResearcherProfile,
        dashboard: DashboardResponse,
        visibility_gap: float,
    ) -> list[PromotionAction]:
        raw = profile.raw_profile
        actions: list[PromotionAction] = []
        if not raw.orcid_id:
            actions.append(
                PromotionAction(
                    title="Connect ORCID to verify publication history",
                    category="profile",
                    priority="high",
                    rationale="ORCID lets the system enrich works, citations, h-index, and topics automatically.",
                    expected_impact=0.86,
                )
            )
        if not raw.google_scholar_url:
            actions.append(
                PromotionAction(
                    title="Add Google Scholar profile link",
                    category="profile",
                    priority="medium",
                    rationale="Google Scholar helps validate citation-oriented visibility outside OpenAlex.",
                    expected_impact=0.58,
                )
            )
        if len(profile.research_keywords) < 3 and dashboard.researcher.topics:
            topics = ", ".join(topic.name for topic in dashboard.researcher.topics[:3])
            actions.append(
                PromotionAction(
                    title=f"Update profile keywords with: {topics}",
                    category="visibility",
                    priority="high",
                    rationale="Better keyword-topic alignment improves discoverability in academic search systems.",
                    expected_impact=0.74,
                )
            )
        if dashboard.opportunities:
            top_source = dashboard.opportunities[0]
            actions.append(
                PromotionAction(
                    title=f"Prepare next paper for {top_source.title}",
                    category="venue",
                    priority="medium",
                    rationale="This venue/source is ranked by semantic match with the profile and recent publications.",
                    expected_impact=max(0.45, top_source.score),
                )
            )
        if dashboard.collaborators:
            top_collaborator = dashboard.collaborators[0]
            actions.append(
                PromotionAction(
                    title=f"Contact {top_collaborator.name} for topic-aligned collaboration",
                    category="collaboration",
                    priority="high" if top_collaborator.score >= 0.5 else "medium",
                    rationale="The candidate was found through real OpenAlex works/authorship signals.",
                    expected_impact=max(0.4, top_collaborator.score),
                )
            )
        if visibility_gap >= 0.45:
            actions.append(
                PromotionAction(
                    title="Publish a preprint or repository version for the next manuscript",
                    category="publication",
                    priority="medium",
                    rationale="Open access artifacts can improve discoverability before formal citations appear.",
                    expected_impact=0.52,
                )
            )
        return sorted(actions, key=lambda item: item.expected_impact, reverse=True)[:6]
