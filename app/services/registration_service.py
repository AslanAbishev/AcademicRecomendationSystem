from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import requests

from app.clients.openalex import OpenAlexClient
from app.schemas import DomainCandidate, RegisteredResearcherProfile, ResearcherRegistrationRequest


DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "Recommender Systems": {"recommendation", "recommender", "ranking", "personalization", "collaborative filtering"},
    "Natural Language Processing": {"nlp", "language model", "text mining", "semantic search", "information extraction"},
    "Information Retrieval": {"information retrieval", "search", "scholarly discovery", "retrieval", "query expansion"},
    "Graph Machine Learning": {"graph", "graphsage", "gnn", "citation network", "co-authorship"},
    "Scientometrics and Research Analytics": {"scientometrics", "citation", "altmetrics", "research evaluation", "bibliometrics"},
    "Cybersecurity": {"cybersecurity", "intrusion", "malware", "threat intelligence", "security analytics"},
    "Computer Vision": {"computer vision", "image", "vision transformer", "object detection", "multimodal"},
    "Data Science and Machine Learning": {"machine learning", "classification", "clustering", "prediction", "analytics"},
    "Digital Libraries and Scholarly Communication": {"digital library", "open science", "scholarly communication", "openalex", "metadata"},
    "Human-Centered AI": {"explainability", "human-computer interaction", "fairness", "trustworthy ai", "user modeling"},
}


def _clean_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(value)
    return cleaned


class RegistrationService:
    def __init__(self, storage_path: Path, client: OpenAlexClient | None = None) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client

    def register(self, payload: ResearcherRegistrationRequest) -> RegisteredResearcherProfile:
        derived_metrics, profile_data_sources, openalex_author, openalex_works = self._derive_metrics(payload)
        profile_values = self._profile_values(payload, openalex_author)
        openalex_topics = self._openalex_topic_names(openalex_author, openalex_works)
        research_keywords = _clean_items(payload.research_keywords) or openalex_topics[:6]
        thematic_clusters = _clean_items(payload.thematic_clusters) or openalex_topics[:4]
        domain_candidates = self._infer_domains(payload, openalex_topics, profile_values)
        profile = RegisteredResearcherProfile(
            profile_id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            full_name=profile_values["full_name"],
            email=payload.email,
            affiliation=profile_values["affiliation"],
            position_title=profile_values["position_title"],
            academic_level=profile_values["academic_level"],
            academic_field=profile_values["academic_field"],
            country=profile_values["country"],
            profile_completeness=self._profile_completeness(
                payload,
                derived_metrics,
                profile_values,
                research_keywords,
                thematic_clusters,
            ),
            visibility_score=self._visibility_score(payload, derived_metrics),
            research_keywords=research_keywords,
            thematic_clusters=thematic_clusters,
            interest_embedding_text=self._interest_embedding_text(
                payload,
                profile_values,
                research_keywords,
                thematic_clusters,
            ),
            domain_candidates=domain_candidates,
            derived_metrics=derived_metrics,
            profile_data_sources=profile_data_sources,
            raw_profile=payload,
        )
        self._append(profile)
        return profile

    def list_profiles(self) -> list[RegisteredResearcherProfile]:
        return [RegisteredResearcherProfile.model_validate(item) for item in self._read_all()]

    def get_profile(self, profile_id: str) -> RegisteredResearcherProfile:
        for item in self._read_all():
            if item.get("profile_id") == profile_id:
                return RegisteredResearcherProfile.model_validate(item)
        raise KeyError(f"Registered profile '{profile_id}' was not found")

    def _append(self, profile: RegisteredResearcherProfile) -> None:
        items = self._read_all()
        items.append(profile.model_dump(mode="json"))
        self.storage_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_all(self) -> list[dict]:
        if not self.storage_path.exists():
            return []
        raw = self.storage_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        return json.loads(raw)

    def _derive_metrics(self, payload: ResearcherRegistrationRequest) -> tuple[dict[str, int | str], list[str], dict | None, list[dict]]:
        metrics = {
            "works_count": int(payload.publication_count),
            "citation_count": int(payload.citation_count),
            "h_index": int(payload.h_index),
            "i10_index": int(payload.i10_index),
            "conference_works_count": int(payload.conference_participation_count),
        }
        sources = ["registration_form"]

        if not self.client or not payload.orcid_id:
            return metrics, sources, None, []

        try:
            author = self.client.get_author_by_orcid(payload.orcid_id)
            works = self.client.get_author_works(author["id"].split("/")[-1], per_page=25)
        except (requests.RequestException, KeyError, ValueError):
            return metrics, sources, None, []

        summary_stats = author.get("summary_stats") or {}
        metrics = {
            "works_count": int(author.get("works_count", metrics["works_count"])),
            "citation_count": int(author.get("cited_by_count", metrics["citation_count"])),
            "h_index": int(summary_stats.get("h_index", metrics["h_index"])),
            "i10_index": int(summary_stats.get("i10_index", metrics["i10_index"])),
            "conference_works_count": self._conference_work_count(works),
        }
        sources = ["registration_form", "openalex_orcid"]
        return metrics, sources, author, works

    def _profile_values(self, payload: ResearcherRegistrationRequest, author: dict | None) -> dict[str, str]:
        affiliation, country = self._openalex_affiliation(author)
        academic_field = payload.academic_field.strip() or self._openalex_primary_topic(author) or "Research"
        return {
            "full_name": payload.full_name.strip() or (author or {}).get("display_name", "Researcher"),
            "affiliation": payload.affiliation.strip() or affiliation or "Unknown affiliation",
            "position_title": payload.position_title.strip() or "Researcher",
            "academic_level": payload.academic_level.strip() or self._infer_academic_level(payload),
            "academic_field": academic_field,
            "country": payload.country.strip() or country or "Unknown country",
        }

    def _openalex_affiliation(self, author: dict | None) -> tuple[str, str]:
        if not author:
            return "", ""
        institutions = author.get("last_known_institutions") or []
        if not institutions:
            return "", ""
        institution = institutions[0]
        return institution.get("display_name", ""), institution.get("country_code", "")

    def _openalex_primary_topic(self, author: dict | None) -> str:
        if not author:
            return ""
        topics = author.get("topics") or []
        if not topics:
            return ""
        return topics[0].get("display_name", "")

    def _openalex_topic_names(self, author: dict | None, works: list[dict], limit: int = 10) -> list[str]:
        seen: set[str] = set()
        topics: list[str] = []
        for topic in (author or {}).get("topics", []):
            name = topic.get("display_name", "").strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                topics.append(name)
        for work in works:
            for topic in work.get("topics", []):
                name = topic.get("display_name", "").strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    topics.append(name)
        return topics[:limit]

    def _infer_academic_level(self, payload: ResearcherRegistrationRequest) -> str:
        if payload.years_in_research <= 5 or payload.publication_count <= 25:
            return "Early-career researcher"
        return "Researcher"

    def _conference_work_count(self, works: list[dict]) -> int:
        count = 0
        for work in works:
            source = (work.get("primary_location") or {}).get("source") or {}
            source_type = source.get("type") or ""
            if source_type == "conference":
                count += 1
        return count

    def _interest_embedding_text(
        self,
        payload: ResearcherRegistrationRequest,
        profile_values: dict[str, str],
        research_keywords: list[str],
        thematic_clusters: list[str],
    ) -> str:
        parts = [
            profile_values["academic_field"],
            profile_values["position_title"],
            profile_values["academic_level"],
            profile_values["affiliation"],
            profile_values["country"],
            " ".join(research_keywords),
            " ".join(thematic_clusters),
            " ".join(_clean_items(payload.methods)),
            " ".join(_clean_items(payload.collaboration_goals)),
            " ".join(_clean_items(payload.publication_goals)),
            payload.bio or "",
        ]
        return " ".join(part for part in parts if part).strip()

    def _infer_domains(
        self,
        payload: ResearcherRegistrationRequest,
        openalex_topics: list[str],
        profile_values: dict[str, str],
    ) -> list[DomainCandidate]:
        profile_terms = " ".join(
            [
                profile_values["academic_field"],
                profile_values["position_title"],
                payload.department or "",
                payload.bio or "",
                " ".join(payload.research_keywords),
                " ".join(payload.thematic_clusters),
                " ".join(openalex_topics),
                " ".join(payload.methods),
                " ".join(payload.target_journals),
                " ".join(payload.collaboration_goals),
                " ".join(payload.publication_goals),
            ]
        ).lower()

        candidates: list[DomainCandidate] = []
        for name, keywords in DOMAIN_KEYWORDS.items():
            matches = sorted({keyword for keyword in keywords if keyword in profile_terms})
            if not matches:
                continue
            score = min(1.0, 0.25 + len(matches) * 0.17)
            candidates.append(DomainCandidate(name=name, score=round(score, 4), matched_terms=matches))

        if not candidates and profile_values["academic_field"].strip():
            candidates.append(
                DomainCandidate(
                    name=profile_values["academic_field"].strip(),
                    score=0.45,
                    matched_terms=[profile_values["academic_field"].strip()],
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:5]

    def _profile_completeness(
        self,
        payload: ResearcherRegistrationRequest,
        derived_metrics: dict[str, int | str],
        profile_values: dict[str, str],
        research_keywords: list[str],
        thematic_clusters: list[str],
    ) -> float:
        checks = [
            bool(profile_values["full_name"].strip()),
            bool(payload.email.strip()),
            bool(profile_values["affiliation"].strip() and profile_values["affiliation"] != "Unknown affiliation"),
            bool(profile_values["position_title"].strip()),
            bool(profile_values["academic_level"].strip()),
            bool(profile_values["academic_field"].strip() and profile_values["academic_field"] != "Research"),
            bool(payload.orcid_id),
            bool(payload.google_scholar_url or payload.researchgate_url),
            bool(research_keywords),
            bool(thematic_clusters),
            bool(_clean_items(payload.publication_goals)),
            bool(payload.bio and payload.bio.strip()),
            int(derived_metrics["works_count"]) > 0,
            int(derived_metrics["conference_works_count"]) > 0,
        ]
        return round(sum(checks) / len(checks), 4)

    def _visibility_score(self, payload: ResearcherRegistrationRequest, derived_metrics: dict[str, int | str]) -> float:
        publication_signal = min(int(derived_metrics["works_count"]) / 20, 1.0)
        citation_signal = min(int(derived_metrics["citation_count"]) / 200, 1.0)
        activity_signal = min(
            (
                int(derived_metrics["conference_works_count"])
                + payload.peer_review_count
            )
            / 20,
            1.0,
        )
        digital_signal = min((payload.profile_views + payload.downloads + payload.altmetric_mentions) / 500, 1.0)
        return round(
            0.35 * publication_signal + 0.2 * citation_signal + 0.25 * activity_signal + 0.2 * digital_signal,
            4,
        )
