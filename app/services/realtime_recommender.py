from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.clients.openalex import OpenAlexClient
from app.config import settings
from app.schemas import (
    CollaboratorRecommendation,
    DashboardAnalytics,
    DashboardResponse,
    OpportunityRecommendation,
    RecentWorkSummary,
    RecommendationExplanation,
    ResearcherProfile,
    ResearcherSearchResult,
    TopicSummary,
)


CURRENT_YEAR = 2026


def _safe_float(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


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


def _topic_names(payload_topics: list[dict], limit: int = 5) -> list[str]:
    return [item.get("display_name", "") for item in payload_topics[:limit] if item.get("display_name")]


def _topic_summaries(payload_topics: list[dict], limit: int = 8) -> list[TopicSummary]:
    return [
        TopicSummary(name=item.get("display_name", ""), score=float(item.get("count", 0)))
        for item in payload_topics[:limit]
        if item.get("display_name")
    ]


class ECRRecommenderService:
    def __init__(self, client: OpenAlexClient | None = None) -> None:
        self.client = client or OpenAlexClient(mailto=settings.openalex_mailto)

    def search_researchers(self, query: str, limit: int = 8) -> list[ResearcherSearchResult]:
        results = self.client.search_authors(query, per_page=limit)
        return [self._map_search_result(item) for item in results]

    def get_dashboard(self, researcher_id: str, top_k: int = 6) -> DashboardResponse:
        author = self.client.get_author(researcher_id)
        works = self.client.get_author_works(researcher_id, per_page=12)
        profile = self._map_profile(author)
        recent_works = [self._map_work_summary(work) for work in works]
        researcher_text = self._build_author_text(author, works)
        opportunities = self._recommend_sources(author, works, researcher_text, top_k=top_k)
        collaborators = self._recommend_collaborators(author, works, researcher_text, top_k=min(top_k, 5))
        analytics = self._build_analytics(author, works)
        return DashboardResponse(
            researcher=profile,
            recent_works=recent_works,
            opportunities=opportunities,
            collaborators=collaborators,
            analytics=analytics,
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

    def _map_work_summary(self, work: dict) -> RecentWorkSummary:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        return RecentWorkSummary(
            work_id=work["id"].split("/")[-1],
            title=work.get("title", "Untitled work"),
            year=work.get("publication_year"),
            cited_by_count=int(work.get("cited_by_count", 0)),
            venue=source.get("display_name"),
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
        source_candidates: dict[str, dict] = {}

        for query in queries:
            for source in self.client.search_sources(query, per_page=6):
                source_candidates[source["id"]] = source

        if not source_candidates:
            return []

        candidate_values = list(source_candidates.values())
        texts = [self._build_source_text(item) for item in candidate_values]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([researcher_text] + texts)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]

        recommendations: list[OpportunityRecommendation] = []
        seen_types: defaultdict[str, int] = defaultdict(int)

        for idx in np.argsort(similarities)[::-1]:
            source = candidate_values[idx]
            source_type = self._map_source_type(source.get("type"))
            relevance = _safe_float(similarities[idx])
            diversity = 1.0 if seen_types[source_type] == 0 else 0.6
            ecr_boost = self._source_ecr_boost(source)
            final_score = self._combine_scores(relevance, diversity, ecr_boost)
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
                        reasons=self._build_source_reasons(source, relevance, ecr_boost),
                    ),
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _recommend_collaborators(
        self,
        author: dict,
        works: list[dict],
        researcher_text: str,
        top_k: int,
    ) -> list[CollaboratorRecommendation]:
        queries = self._build_topic_queries(author, works)
        candidate_map: dict[str, dict] = {}

        for query in queries:
            for candidate in self.client.search_authors(query, per_page=8):
                candidate_id = candidate["id"].split("/")[-1]
                if candidate_id == author["id"].split("/")[-1]:
                    continue
                candidate_map[candidate_id] = candidate

        if not candidate_map:
            return []

        candidate_values = list(candidate_map.values())
        texts = [self._build_author_text(item, []) for item in candidate_values]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([researcher_text] + texts)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]

        recommendations: list[CollaboratorRecommendation] = []

        for idx in np.argsort(similarities)[::-1]:
            candidate = candidate_values[idx]
            relevance = _safe_float(similarities[idx])
            diversity = self._collaborator_diversity(author, candidate)
            ecr_boost = self._collaborator_ecr_boost(candidate)
            final_score = self._combine_scores(relevance, diversity, ecr_boost)
            recommendations.append(
                CollaboratorRecommendation(
                    researcher_id=candidate["id"].split("/")[-1],
                    name=candidate.get("display_name", "Unknown author"),
                    affiliation=_extract_affiliation(candidate),
                    country_code=_extract_country(candidate),
                    score=final_score,
                    explanation=RecommendationExplanation(
                        relevance=relevance,
                        diversity=diversity,
                        ecr_boost=ecr_boost,
                        final_score=final_score,
                        reasons=self._build_collaborator_reasons(author, candidate, diversity, ecr_boost),
                    ),
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

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
                source.get("display_name", ""),
                source.get("type", ""),
                " ".join(_topic_names(source.get("topics", []), limit=8)),
                source.get("host_organization_name", ""),
                source.get("country_code", ""),
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

    def _build_collaborator_reasons(self, author: dict, candidate: dict, diversity: float, ecr_boost: float) -> list[str]:
        reasons = []
        shared = sorted(set(_topic_names(author.get("topics", []), 6)) & set(_topic_names(candidate.get("topics", []), 6)))
        if shared:
            reasons.append(f"Shared topics: {', '.join(shared[:3])}")
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

    def _collaborator_diversity(self, author: dict, candidate: dict) -> float:
        same_institution = _extract_affiliation(author) == _extract_affiliation(candidate)
        same_country = _extract_country(author) == _extract_country(candidate)
        return _safe_float(1.0 - (0.45 if same_institution else 0.0) - (0.2 if same_country else 0.0))

    def _collaborator_ecr_boost(self, candidate: dict) -> float:
        works_count = int(candidate.get("works_count", 0))
        return _safe_float(0.85 if works_count <= 25 else 0.35)

    def _combine_scores(self, relevance: float, diversity: float, ecr_boost: float) -> float:
        final_score = (
            settings.relevance_weight * relevance
            + settings.diversity_weight * diversity
            + settings.ecr_weight * ecr_boost
        )
        return round(_safe_float(final_score), 4)
