from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.data_loader import (
    load_openalex_researchers,
    load_openalex_works,
    load_opportunities,
    load_researchers,
)
from app.schemas import (
    CollaboratorRecommendation,
    DashboardAnalytics,
    DashboardResponse,
    OpportunityCatalogItem,
    OpportunityRecommendation,
    RecommendationExplanation,
    ResearcherSummary,
)


CURRENT_YEAR = 2026


def _split_tags(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split("|") if item.strip()]


def _safe_float(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class VectorSpace:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray


class ECRRecommenderService:
    def __init__(self) -> None:
        self.researchers = load_researchers().copy()
        self.opportunities = load_opportunities().copy()
        self.openalex_researchers = load_openalex_researchers().copy()
        self.openalex_works = load_openalex_works().copy()
        self.researchers["external_context"] = self.researchers.apply(self._build_external_context, axis=1)
        self.researchers["profile_text"] = self.researchers.apply(self._build_researcher_text, axis=1)
        self.opportunities["content_text"] = self.opportunities.apply(self._build_opportunity_text, axis=1)
        self.profile_space = self._fit_vector_space(
            self.researchers["profile_text"].tolist() + self.opportunities["content_text"].tolist()
        )

    def list_researchers(self) -> list[ResearcherSummary]:
        records = self.researchers.to_dict(orient="records")
        return [ResearcherSummary(**record) for record in records]

    def list_opportunities(self) -> list[OpportunityCatalogItem]:
        records = self.opportunities.to_dict(orient="records")
        return [
            OpportunityCatalogItem(
                **{
                    **record,
                    "ecr_friendly": str(record["ecr_friendly"]).lower() == "true",
                }
            )
            for record in records
        ]

    def get_dashboard(self, researcher_id: int, top_k: int = 6) -> DashboardResponse:
        researcher = self._get_researcher_row(researcher_id)
        opportunities = self.recommend_opportunities(researcher_id, top_k=top_k)
        collaborators = self.recommend_collaborators(researcher_id, top_k=min(top_k, 4))
        analytics = self._build_analytics(researcher)
        summary = ResearcherSummary(**researcher.to_dict())
        return DashboardResponse(
            researcher=summary,
            opportunities=opportunities,
            collaborators=collaborators,
            analytics=analytics,
        )

    def recommend_opportunities(self, researcher_id: int, top_k: int = 6) -> list[OpportunityRecommendation]:
        researcher = self._get_researcher_row(researcher_id)
        profile_vector = self.profile_space.vectorizer.transform([researcher["profile_text"]])
        opportunity_vectors = self.profile_space.vectorizer.transform(self.opportunities["content_text"])
        similarities = cosine_similarity(profile_vector, opportunity_vectors)[0]

        recommendations: list[OpportunityRecommendation] = []
        seen_types: dict[str, int] = {}

        for idx in np.argsort(similarities)[::-1]:
            row = self.opportunities.iloc[idx]
            relevance = _safe_float(similarities[idx])
            diversity = self._type_diversity_bonus(row["opportunity_type"], seen_types)
            ecr_boost = self._opportunity_ecr_boost(researcher, row)
            final_score = self._combine_scores(relevance, diversity, ecr_boost)
            explanation = RecommendationExplanation(
                relevance=relevance,
                diversity=diversity,
                ecr_boost=ecr_boost,
                final_score=final_score,
                reasons=self._build_opportunity_reasons(researcher, row, relevance, ecr_boost),
            )
            recommendations.append(
                OpportunityRecommendation(
                    opportunity_id=row["opportunity_id"],
                    opportunity_type=row["opportunity_type"],
                    title=row["title"],
                    score=final_score,
                    region=row["region"],
                    deadline=row["deadline"],
                    description=row["description"],
                    explanation=explanation,
                )
            )
            seen_types[row["opportunity_type"]] = seen_types.get(row["opportunity_type"], 0) + 1
            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_collaborators(self, researcher_id: int, top_k: int = 4) -> list[CollaboratorRecommendation]:
        researcher = self._get_researcher_row(researcher_id)
        candidate_df = self.researchers[self.researchers["researcher_id"] != researcher_id].copy()
        profile_vector = self.profile_space.vectorizer.transform([researcher["profile_text"]])
        candidate_vectors = self.profile_space.vectorizer.transform(candidate_df["profile_text"])
        similarities = cosine_similarity(profile_vector, candidate_vectors)[0]

        recommendations: list[CollaboratorRecommendation] = []

        for local_idx, score in sorted(enumerate(similarities), key=lambda item: item[1], reverse=True):
            candidate = candidate_df.iloc[local_idx]
            relevance = _safe_float(score)
            diversity = self._complementarity_score(researcher, candidate)
            ecr_boost = self._candidate_ecr_boost(researcher, candidate)
            final_score = self._combine_scores(relevance, diversity, ecr_boost)
            explanation = RecommendationExplanation(
                relevance=relevance,
                diversity=diversity,
                ecr_boost=ecr_boost,
                final_score=final_score,
                reasons=self._build_collaborator_reasons(researcher, candidate, diversity, ecr_boost),
            )
            recommendations.append(
                CollaboratorRecommendation(
                    researcher_id=int(candidate["researcher_id"]),
                    name=str(candidate["name"]),
                    domain=str(candidate["domain"]),
                    score=final_score,
                    explanation=explanation,
                )
            )
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _fit_vector_space(self, corpus: list[str]) -> VectorSpace:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
        return VectorSpace(vectorizer=vectorizer, matrix=matrix)

    def _build_researcher_text(self, row: pd.Series) -> str:
        publication_bucket = self._publication_bucket(int(row["publication_count"]))
        career_stage_signal = "early-career researcher" if self._is_ecr(int(row["phd_year"])) else "established researcher"
        return " ".join(
            [
                str(row["domain"]),
                str(row["keywords"]),
                str(row["skills"]),
                str(row["languages"]),
                str(row["goals"]),
                str(row["region"]),
                publication_bucket,
                career_stage_signal,
                str(row.get("external_context", "")),
            ]
        )

    def _build_opportunity_text(self, row: pd.Series) -> str:
        return " ".join(
            [
                str(row["title"]),
                str(row["description"]),
                str(row["keywords"]),
                str(row["target_domains"]),
                str(row["region"]),
            ]
        )

    def _combine_scores(self, relevance: float, diversity: float, ecr_boost: float) -> float:
        final_score = (
            settings.relevance_weight * relevance
            + settings.diversity_weight * diversity
            + settings.ecr_weight * ecr_boost
        )
        return round(_safe_float(final_score), 4)

    def _get_researcher_row(self, researcher_id: int) -> pd.Series:
        matches = self.researchers[self.researchers["researcher_id"] == researcher_id]
        if matches.empty:
            raise KeyError(f"Researcher {researcher_id} not found")
        return matches.iloc[0]

    def _is_ecr(self, phd_year: int) -> bool:
        return CURRENT_YEAR - int(phd_year) <= 5

    def _publication_bucket(self, publication_count: int) -> str:
        if publication_count <= 2:
            return "cold-start profile"
        if publication_count <= 7:
            return "emerging publication record"
        if publication_count <= 15:
            return "growing publication record"
        return "established publication record"

    def _type_diversity_bonus(self, opportunity_type: str, seen_types: dict[str, int]) -> float:
        return 1.0 if seen_types.get(opportunity_type, 0) == 0 else 0.55

    def _opportunity_ecr_boost(self, researcher: pd.Series, opportunity: pd.Series) -> float:
        boost = 0.0
        if self._is_ecr(int(researcher["phd_year"])):
            boost += 0.5
        if str(opportunity["ecr_friendly"]).lower() == "true":
            boost += 0.5
        return _safe_float(boost)

    def _candidate_ecr_boost(self, researcher: pd.Series, candidate: pd.Series) -> float:
        same_stage = self._is_ecr(int(researcher["phd_year"])) == self._is_ecr(int(candidate["phd_year"]))
        regional_bonus = str(researcher["region"]).lower() == str(candidate["region"]).lower()
        return _safe_float((0.6 if same_stage else 0.25) + (0.4 if regional_bonus else 0.0))

    def _complementarity_score(self, researcher: pd.Series, candidate: pd.Series) -> float:
        left = set(_split_tags(researcher["skills"]))
        right = set(_split_tags(candidate["skills"]))
        if not left or not right:
            return 0.4
        overlap = len(left & right)
        unique_total = len(left | right)
        return _safe_float(1 - (overlap / max(unique_total, 1)))

    def _build_opportunity_reasons(
        self,
        researcher: pd.Series,
        opportunity: pd.Series,
        relevance: float,
        ecr_boost: float,
    ) -> list[str]:
        reasons = []
        researcher_keywords = set(_split_tags(researcher["keywords"]))
        opportunity_keywords = set(_split_tags(opportunity["keywords"]))
        overlap = sorted(researcher_keywords & opportunity_keywords)
        if overlap:
            reasons.append(f"Keyword overlap: {', '.join(overlap[:3])}")
        if relevance >= 0.35:
            reasons.append("High semantic fit between profile and opportunity description")
        if ecr_boost >= 0.8:
            reasons.append("Opportunity is especially suitable for early-career researchers")
        if str(researcher["region"]).lower() == str(opportunity["region"]).lower():
            reasons.append("Regional proximity can simplify networking and applications")
        return reasons or ["General profile match based on domain and research goals"]

    def _build_collaborator_reasons(
        self,
        researcher: pd.Series,
        candidate: pd.Series,
        diversity: float,
        ecr_boost: float,
    ) -> list[str]:
        reasons = []
        researcher_keywords = set(_split_tags(researcher["keywords"]))
        candidate_keywords = set(_split_tags(candidate["keywords"]))
        overlap = sorted(researcher_keywords & candidate_keywords)
        if overlap:
            reasons.append(f"Shared research topics: {', '.join(overlap[:3])}")
        if diversity >= 0.55:
            reasons.append("Complementary skill mix can improve collaboration outcomes")
        if ecr_boost >= 0.6:
            reasons.append("Career-stage alignment helps reduce cold-start mismatch")
        return reasons or ["Balanced collaborator fit from content and profile similarity"]

    def _build_analytics(self, researcher: pd.Series) -> DashboardAnalytics:
        publication_count = int(researcher["publication_count"])
        h_index = int(researcher["h_index"])
        ecr_status = self._is_ecr(int(researcher["phd_year"]))
        profile_strength = _safe_float((publication_count / 12) * 0.5 + (h_index / 10) * 0.5)
        cold_start_risk = _safe_float(1.0 - ((publication_count / 10) * 0.7 + (h_index / 8) * 0.3))
        collaboration_readiness = _safe_float(
            0.4 + min(len(_split_tags(researcher["skills"])), 6) / 10 + (0.15 if ecr_status else 0.0)
        )
        return DashboardAnalytics(
            profile_strength=round(profile_strength, 4),
            cold_start_risk=round(cold_start_risk, 4),
            ecr_status=ecr_status,
            publication_count=publication_count,
            collaboration_readiness=round(collaboration_readiness, 4),
        )

    def _build_external_context(self, researcher: pd.Series) -> str:
        fragments: list[str] = []

        author_match = self._match_openalex_author_context(researcher)
        if author_match:
            fragments.append(author_match)

        work_match = self._match_openalex_work_context(researcher)
        if work_match:
            fragments.append(work_match)

        return " ".join(fragments)

    def _match_openalex_author_context(self, researcher: pd.Series) -> str:
        if self.openalex_researchers.empty:
            return ""

        keywords = set(_split_tags(researcher["keywords"]))
        ranked: list[tuple[int, pd.Series]] = []
        for _, candidate in self.openalex_researchers.iterrows():
            candidate_topics = set(_split_tags(candidate.get("topics", "")))
            overlap = len(keywords & candidate_topics)
            if overlap:
                ranked.append((overlap, candidate))

        if not ranked:
            return ""

        ranked.sort(key=lambda item: (item[0], item[1].get("works_count", 0)), reverse=True)
        best = ranked[0][1]
        return (
            f"openalex_author_context {best.get('name', '')} "
            f"{best.get('topics', '')} {best.get('institutions', '')}"
        )

    def _match_openalex_work_context(self, researcher: pd.Series) -> str:
        if self.openalex_works.empty:
            return ""

        keywords = set(_split_tags(researcher["keywords"]))
        ranked: list[tuple[int, pd.Series]] = []
        for _, work in self.openalex_works.iterrows():
            work_topics = set(_split_tags(work.get("topics", "")))
            overlap = len(keywords & work_topics)
            if overlap:
                ranked.append((overlap, work))

        if not ranked:
            return ""

        ranked.sort(key=lambda item: (item[0], item[1].get("cited_by_count", 0)), reverse=True)
        best = ranked[0][1]
        return (
            f"openalex_work_context {best.get('title', '')} "
            f"{best.get('topics', '')} {best.get('abstract_excerpt', '')}"
        )
