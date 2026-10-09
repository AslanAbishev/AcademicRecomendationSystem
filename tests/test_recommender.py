import numpy as np

from app.services.realtime_recommender import ECRRecommenderService
from app.schemas import RegisteredResearcherProfile, ResearcherRegistrationRequest


class FakeOpenAlexClient:
    def search_authors(self, query: str | None = None, per_page: int = 8, extra_filters=None, sort=None):
        if extra_filters:
            return []
        if "Recommender Systems" in (query or ""):
            return [
                {
                    "id": "https://openalex.org/A123",
                    "display_name": "Alice Researcher",
                    "works_count": 12,
                    "cited_by_count": 120,
                    "last_known_institutions": [{"display_name": "Test University", "country_code": "KZ"}],
                    "topics": [{"display_name": "Recommender Systems", "count": 10}],
                },
                {
                    "id": "https://openalex.org/A456",
                    "display_name": "Bob Collaborator",
                    "works_count": 9,
                    "cited_by_count": 40,
                    "last_known_institutions": [{"display_name": "Another Lab", "country_code": "DE"}],
                    "topics": [{"display_name": "Recommender Systems", "count": 8}],
                },
            ]
        return [
            {
                "id": "https://openalex.org/A456",
                "display_name": "Bob Collaborator",
                "works_count": 9,
                "cited_by_count": 40,
                "last_known_institutions": [{"display_name": "Another Lab", "country_code": "DE"}],
                "topics": [{"display_name": "Fairness", "count": 5}],
            }
        ]

    def get_author(self, author_id: str):
        return {
            "id": f"https://openalex.org/{author_id}",
            "display_name": "Alice Researcher",
            "works_count": 12,
            "cited_by_count": 120,
            "summary_stats": {"h_index": 8},
            "last_known_institutions": [{"display_name": "Test University", "country_code": "KZ"}],
            "topics": [{"display_name": "Recommender Systems", "count": 10}],
            "ids": {"orcid": "https://orcid.org/0000-0000-0000-0000"},
            "homepage_url": "https://example.org/alice",
        }

    def get_author_by_orcid(self, orcid: str):
        return self.get_author("A123")

    def get_author_works(self, author_id: str, per_page: int = 12):
        return [
            {
                "id": "https://openalex.org/W1",
                "title": "Fair ranking for scholarly discovery",
                "publication_year": 2024,
                "cited_by_count": 15,
                "topics": [{"display_name": "Recommender Systems"}, {"display_name": "Fairness"}],
                "primary_location": {"source": {"display_name": "Information Retrieval"}},
            }
        ]

    def search_works(self, query: str, per_page: int = 12, extra_filters=None):
        return self._works()

    def search_institutions(self, query: str, per_page: int = 5):
        return [
            {"id": "https://openalex.org/I123", "display_name": "Test University Research Center"},
        ]

    def _works(self):
        return [
            {
                "id": "https://openalex.org/W2",
                "title": "Collaborative scholarly recommendation with fairness constraints",
                "publication_year": 2025,
                "cited_by_count": 6,
                "topics": [{"display_name": "Recommender Systems"}, {"display_name": "Digital Libraries"}],
                "primary_location": {"source": {"display_name": "Journal of Scholarly Discovery", "type": "journal"}},
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A123", "display_name": "Alice Researcher"},
                        "institutions": [{"display_name": "Test University", "country_code": "KZ"}],
                        "countries": ["KZ"],
                    },
                    {
                        "author": {"id": "https://openalex.org/A456", "display_name": "Bob Collaborator"},
                        "institutions": [{"display_name": "Another Lab", "country_code": "DE"}],
                        "countries": ["DE"],
                    },
                    {
                        "author": {"id": "https://openalex.org/A789", "display_name": "Carol Local"},
                        "institutions": [{"display_name": "Test University Research Center", "country_code": "KZ"}],
                        "countries": ["KZ"],
                    },
                ],
            },
            {
                "id": "https://openalex.org/W3",
                "title": "Clinical surgery case notes",
                "publication_year": 2025,
                "cited_by_count": 1,
                "topics": [{"display_name": "Surgery"}, {"display_name": "Anatomy"}],
                "primary_location": {"source": {"display_name": "Medical Notes", "type": "journal"}},
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A999", "display_name": "Mallory Offtopic"},
                        "institutions": [{"display_name": "Unrelated Lab", "country_code": "US"}],
                        "countries": ["US"],
                    },
                ],
            },
        ]

    def search_sources(self, query: str, per_page: int = 8):
        return [
            {
                "id": "https://openalex.org/S1",
                "display_name": "Information Retrieval",
                "type": "journal",
                "country_code": "NL",
                "homepage_url": "https://example.org/ir",
                "is_oa": True,
                "is_in_doaj": False,
                "works_count": 696,
                "summary_stats": {"h_index": 79},
                "topics": [{"display_name": "Recommender Systems"}, {"display_name": "Information Retrieval"}],
                "host_organization_name": "Springer",
            }
        ]


class FakeOpenAlexClientWithDifferentOpenAlexCountry(FakeOpenAlexClient):
    def get_author(self, author_id: str):
        author = super().get_author(author_id)
        author["last_known_institutions"] = [{"display_name": "Remote University", "country_code": "US"}]
        return author


class FakeOpenAlexClientWithScopedWorks(FakeOpenAlexClient):
    def search_works(self, query: str, per_page: int = 12, extra_filters=None):
        filters = extra_filters or []
        works = self._works()
        if "institutions.country_code:KZ" in filters:
            return [
                work
                for work in works
                if any("KZ" in authorship.get("countries", []) for authorship in work.get("authorships", []))
            ]
        if any(filter_value.startswith("institutions.id:") for filter_value in filters):
            return works[:1]
        return works


class FakeOpenAlexClientWithAffiliationAuthorFallback(FakeOpenAlexClient):
    def search_works(self, query: str, per_page: int = 12, extra_filters=None):
        filters = extra_filters or []
        if any(filter_value.startswith("institutions.id:") for filter_value in filters):
            return []
        if any(filter_value.startswith("raw_affiliation_strings.search:") for filter_value in filters):
            return []
        return super().search_works(query, per_page=per_page, extra_filters=extra_filters)

    def search_authors(self, query: str | None = None, per_page: int = 8, extra_filters=None, sort=None):
        filters = extra_filters or []
        if "last_known_institutions.id:https://openalex.org/I123" not in filters:
            return super().search_authors(query, per_page=per_page, extra_filters=extra_filters, sort=sort)
        if any(filter_value.startswith("topics.id:") for filter_value in filters):
            return []
        return [
            {
                "id": "https://openalex.org/A777",
                "display_name": "Dana Same Institution",
                "works_count": 14,
                "cited_by_count": 25,
                "last_known_institutions": [{"display_name": "Test University Research Center", "country_code": "KZ"}],
                "topics": [{"display_name": "Recommender Systems", "count": 5}],
            }
        ]


class FakeOpenAlexClientWithRawAffiliationFallback(FakeOpenAlexClient):
    def search_works(self, query: str, per_page: int = 12, extra_filters=None):
        filters = extra_filters or []
        if any(filter_value.startswith("institutions.id:") for filter_value in filters):
            return []
        if any(filter_value.startswith("raw_affiliation_strings.search:") for filter_value in filters):
            return [
                {
                    "id": "https://openalex.org/W4",
                    "title": "Computer vision for digital education labs",
                    "publication_year": 2025,
                    "cited_by_count": 4,
                    "topics": [{"display_name": "Computer Vision"}, {"display_name": "Educational Technology"}],
                    "primary_location": {"source": {"display_name": "Digital Education Journal", "type": "journal"}},
                    "authorships": [
                        {
                            "author": {"id": "https://openalex.org/A888", "display_name": "Erlan AITU"},
                            "institutions": [{"display_name": "Astana Medical University", "country_code": "KZ"}],
                            "countries": ["KZ"],
                            "raw_affiliation_strings": [
                                "Department of Intelligent Systems and Cybersecurity, Astana IT University, Kazakhstan"
                            ],
                        },
                        {
                            "author": {"id": "https://openalex.org/A999", "display_name": "Outside Coauthor"},
                            "institutions": [{"display_name": "Outside University", "country_code": "US"}],
                            "countries": ["US"],
                            "raw_affiliation_strings": ["Outside University, United States"],
                        },
                    ],
                }
            ]
        return []


class FakeResearchDatabase:
    def search_similar_works(self, embedding, model_name="hashing", limit=10, exclude_work_ids=None):
        return [
            {
                "work_id": "W900",
                "title": "SPECTER2 recommendations for scholarly visibility",
                "abstract": "Transformer embeddings for academic recommender systems.",
                "publication_year": 2025,
                "cited_by_count": 12,
                "source_name": "Journal of Scholarly Discovery",
                "source_type": "journal",
                "landing_page_url": "https://example.org/work",
                "topics": ["Recommender Systems", "Digital Libraries"],
                "similarity": 0.82,
            }
        ]


def test_search_researchers_returns_real_author_shape():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    results = service.search_researchers("Alice")

    assert results[0].researcher_id
    assert results[0].name


def test_dashboard_contains_real_time_profile_and_work():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    dashboard = service.get_dashboard("A123", top_k=3)

    assert dashboard.researcher.researcher_id == "A123"
    assert dashboard.recent_works[0].title == "Fair ranking for scholarly discovery"
    assert dashboard.analytics.publication_count == 12


def test_collaborator_scores_exclude_self():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    dashboard = service.get_dashboard("A123", top_k=3)

    ids = [item.researcher_id for item in dashboard.collaborators]
    assert "A123" not in ids
    assert "A999" not in ids
    assert dashboard.collaborators[0].evidence_works[0].title == "Collaborative scholarly recommendation with fairness constraints"


def test_collaborator_scope_filters_by_country_and_affiliation():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    global_dashboard = service.get_dashboard("A123", top_k=3, collaborator_scope="global")
    same_country = service.get_dashboard("A123", top_k=3, collaborator_scope="same_country")
    same_affiliation = service.get_dashboard("A123", top_k=3, collaborator_scope="same_affiliation")

    assert {item.researcher_id for item in global_dashboard.collaborators} != {
        item.researcher_id for item in same_country.collaborators
    }
    assert {item.country_code for item in same_country.collaborators} == {"KZ"}
    assert {item.affiliation for item in same_affiliation.collaborators} == {"Test University Research Center"}
    assert same_country.collaborator_scope == "same_country"
    assert all("same country" in item.scope_match for item in same_country.collaborators)


def test_opportunities_are_real_sources_not_fake_catalog_entries():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    dashboard = service.get_dashboard("A123", top_k=3)

    assert dashboard.opportunities[0].title == "Information Retrieval"
    assert dashboard.opportunities[0].opportunity_type == "journal"


def test_user_cabinet_contains_connected_profiles_and_dashboard():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    profile = RegisteredResearcherProfile(
        profile_id="profile-1",
        created_at="2026-04-29T10:00:00+00:00",
        full_name="Alice Researcher",
        email="alice@example.com",
        affiliation="Test University",
        position_title="PhD student",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        profile_completeness=0.8,
        visibility_score=0.5,
        research_keywords=["recommender systems"],
        thematic_clusters=["digital libraries"],
        interest_embedding_text="computer science recommender systems",
        domain_candidates=[],
        derived_metrics={"works_count": 12, "citation_count": 120, "h_index": 8, "i10_index": 5, "conference_works_count": 1},
        profile_data_sources=["registration_form", "openalex_orcid"],
        raw_profile=ResearcherRegistrationRequest(
            full_name="Alice Researcher",
            email="alice@example.com",
            orcid_id="0000-0000-0000-0000",
            affiliation="Test University",
            position_title="PhD student",
            academic_level="ECR",
            academic_field="Computer Science",
            country="Kazakhstan",
            google_scholar_url="https://scholar.google.com/citations?user=test",
        ),
    )

    cabinet = service.get_user_cabinet(profile, top_k=3)

    assert cabinet.connected_profiles[0].platform == "ORCID"
    assert cabinet.personalized_dashboard.recent_works[0].title == "Fair ranking for scholarly discovery"


def test_registered_profile_scope_uses_registration_country_even_with_orcid():
    service = ECRRecommenderService(client=FakeOpenAlexClientWithDifferentOpenAlexCountry())
    profile = RegisteredResearcherProfile(
        profile_id="profile-2",
        created_at="2026-04-29T10:00:00+00:00",
        full_name="Alice Researcher",
        email="alice@example.com",
        affiliation="Test University",
        position_title="PhD student",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        profile_completeness=0.8,
        visibility_score=0.5,
        research_keywords=["recommender systems"],
        thematic_clusters=["digital libraries"],
        interest_embedding_text="computer science recommender systems",
        domain_candidates=[],
        derived_metrics={"works_count": 12, "citation_count": 120, "h_index": 8, "i10_index": 5, "conference_works_count": 1},
        profile_data_sources=["registration_form", "openalex_orcid"],
        raw_profile=ResearcherRegistrationRequest(
            full_name="Alice Researcher",
            email="alice@example.com",
            orcid_id="0000-0000-0000-0000",
            affiliation="Test University",
            position_title="PhD student",
            academic_level="ECR",
            academic_field="Computer Science",
            country="Kazakhstan",
        ),
    )

    cabinet = service.get_user_cabinet(profile, top_k=3, collaborator_scope="same_country")

    assert cabinet.personalized_dashboard.collaborator_scope == "same_country"
    assert {item.country_code for item in cabinet.personalized_dashboard.collaborators} == {"KZ"}


def test_registered_orcid_dashboard_preserves_registered_affiliation():
    service = ECRRecommenderService(client=FakeOpenAlexClientWithDifferentOpenAlexCountry())
    profile = RegisteredResearcherProfile(
        profile_id="profile-3",
        created_at="2026-04-29T10:00:00+00:00",
        full_name="Talgat Sembayev",
        email="talgat@example.com",
        affiliation="Astana IT University",
        position_title="Assistant Professor",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        profile_completeness=0.8,
        visibility_score=0.5,
        research_keywords=["computer vision"],
        thematic_clusters=["VR AR education"],
        interest_embedding_text="computer vision VR AR education",
        domain_candidates=[],
        derived_metrics={"works_count": 12, "citation_count": 120, "h_index": 8, "i10_index": 5, "conference_works_count": 1},
        profile_data_sources=["registration_form", "openalex_orcid"],
        raw_profile=ResearcherRegistrationRequest(
            full_name="Talgat Sembayev",
            email="talgat@example.com",
            orcid_id="0000-0003-2360-8767",
            affiliation="Astana IT University",
            position_title="Assistant Professor",
            academic_level="ECR",
            academic_field="Computer Science",
            country="Kazakhstan",
        ),
    )

    cabinet = service.get_user_cabinet(profile, top_k=3)

    assert cabinet.personalized_dashboard.researcher.affiliation == "Astana IT University"
    assert cabinet.personalized_dashboard.researcher.country_code == "KZ"


def test_same_scope_uses_openalex_filters_before_candidate_extraction():
    service = ECRRecommenderService(client=FakeOpenAlexClientWithScopedWorks())
    same_country = service.get_dashboard("A123", top_k=3, collaborator_scope="same_country")
    same_affiliation = service.get_dashboard("A123", top_k=3, collaborator_scope="same_affiliation")

    assert same_country.collaborators
    assert same_affiliation.collaborators
    assert all(item.country_code == "KZ" for item in same_country.collaborators)
    assert all("Test University" in item.affiliation for item in same_affiliation.collaborators)


def test_same_affiliation_falls_back_to_author_profiles_when_work_search_is_empty():
    service = ECRRecommenderService(client=FakeOpenAlexClientWithAffiliationAuthorFallback())
    dashboard = service.get_dashboard("A123", top_k=3, collaborator_scope="same_affiliation")

    assert dashboard.collaborators
    assert dashboard.collaborators[0].researcher_id == "A777"
    assert "Test University" in dashboard.collaborators[0].affiliation
    assert dashboard.collaborators[0].scope_match == "same affiliation: Test University"


def test_same_affiliation_uses_raw_affiliation_strings_when_openalex_maps_institution_wrongly():
    service = ECRRecommenderService(client=FakeOpenAlexClientWithRawAffiliationFallback())
    profile = RegisteredResearcherProfile(
        profile_id="profile-4",
        created_at="2026-04-29T10:00:00+00:00",
        full_name="Talgat Sembayev",
        email="talgat@example.com",
        affiliation="Astana IT University",
        position_title="Assistant Professor",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        profile_completeness=0.8,
        visibility_score=0.5,
        research_keywords=["computer vision"],
        thematic_clusters=["digital education"],
        interest_embedding_text="computer vision digital education",
        domain_candidates=[],
        derived_metrics={"works_count": 12, "citation_count": 120, "h_index": 8, "i10_index": 5, "conference_works_count": 1},
        profile_data_sources=["registration_form", "openalex_orcid"],
        raw_profile=ResearcherRegistrationRequest(
            full_name="Talgat Sembayev",
            email="talgat@example.com",
            orcid_id="0000-0003-2360-8767",
            affiliation="Astana IT University",
            position_title="Assistant Professor",
            academic_level="ECR",
            academic_field="Computer Science",
            country="Kazakhstan",
        ),
    )

    cabinet = service.get_user_cabinet(profile, top_k=3, collaborator_scope="same_affiliation")

    assert cabinet.personalized_dashboard.collaborators
    assert cabinet.personalized_dashboard.collaborators[0].researcher_id == "A888"
    assert cabinet.personalized_dashboard.collaborators[0].affiliation == "Astana IT University"
    assert {item.country_code for item in cabinet.personalized_dashboard.collaborators} == {"KZ"}


def test_user_cabinet_contains_visibility_twin_gap_model_and_roadmap():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    profile = RegisteredResearcherProfile(
        profile_id="profile-5",
        created_at="2026-04-29T10:00:00+00:00",
        full_name="Alice Researcher",
        email="alice@example.com",
        affiliation="Test University",
        position_title="PhD student",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        profile_completeness=0.7,
        visibility_score=0.5,
        research_keywords=["recommender systems"],
        thematic_clusters=["digital libraries"],
        interest_embedding_text="computer science recommender systems digital libraries",
        domain_candidates=[],
        derived_metrics={"works_count": 12, "citation_count": 120, "h_index": 8, "i10_index": 5, "conference_works_count": 1},
        profile_data_sources=["registration_form", "openalex_orcid"],
        raw_profile=ResearcherRegistrationRequest(
            full_name="Alice Researcher",
            email="alice@example.com",
            orcid_id="0000-0000-0000-0000",
            affiliation="Test University",
            position_title="PhD student",
            academic_level="ECR",
            academic_field="Computer Science",
            country="Kazakhstan",
        ),
    )

    cabinet = service.get_user_cabinet(profile, top_k=3)

    assert "Visibility Twin compares" in cabinet.promotion_tracker.visibility_twin_summary
    assert len(cabinet.promotion_tracker.gap_dimensions) == 5
    assert {step.phase for step in cabinet.promotion_tracker.roadmap} >= {"Week 1", "Month 1"}
    assert {signal.name for signal in cabinet.promotion_tracker.ml_signals} >= {
        "Model comparison layer",
        "Transformer embeddings",
        "Graph collaboration layer",
        "Fairness-aware reranking",
    }
    assert cabinet.promotion_tracker.recommended_actions


def test_collaborator_explanations_include_hybrid_graph_and_fairness_signals():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    dashboard = service.get_dashboard("A123", top_k=3)

    reasons = " ".join(dashboard.collaborators[0].explanation.reasons)
    model_names = {item.name for item in dashboard.collaborators[0].explanation.model_scores}

    assert "Model comparison" in reasons
    assert "Graph collaboration evidence score" in reasons
    assert "Fairness-aware ECR exposure bonus" in reasons
    assert model_names >= {"TF-IDF cosine", "BM25", "LSA/SVD", "Semantic hashing", "Hybrid ensemble"}


def test_live_collaborator_ranking_uses_graphsage_embeddings():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    service._graphsage_ranker.embeddings = {
        "A123": np.array([1.0, 0.0], dtype="float32"),
        "A456": np.array([1.0, 0.0], dtype="float32"),
        "A789": np.array([1.0, 0.0], dtype="float32"),
    }

    dashboard = service.get_dashboard("A123", top_k=3)

    reasons = " ".join(dashboard.collaborators[0].explanation.reasons)
    model_names = {item.name for item in dashboard.collaborators[0].explanation.model_scores}
    assert "GraphSAGE author-embedding similarity" in reasons
    assert "GraphSAGE" in model_names


def test_live_collaborator_ranking_uses_selected_transformer_embedding():
    class FakeTransformer:
        def encode(self, texts):
            return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), (len(texts), 1))

    service = ECRRecommenderService(client=FakeOpenAlexClient())
    service._embedding_models["specter2"] = FakeTransformer()

    dashboard = service.get_dashboard("A123", top_k=3, embedding_model="specter2")

    reasons = " ".join(dashboard.collaborators[0].explanation.reasons)
    model_names = {item.name for item in dashboard.collaborators[0].explanation.model_scores}
    assert "SPECTER2 collaborator semantic similarity" in reasons
    assert "SPECTER2" in model_names


def test_dashboard_includes_pgvector_paper_recommendations_for_selected_model():
    service = ECRRecommenderService(client=FakeOpenAlexClient(), research_db=FakeResearchDatabase())
    dashboard = service.get_dashboard("A123", top_k=3, embedding_model="hashing")

    assert dashboard.embedding_model == "hashing"
    assert "pgvector" in dashboard.embedding_model_status
    assert dashboard.recommended_works[0].opportunity_type == "paper"
    assert dashboard.recommended_works[0].title == "SPECTER2 recommendations for scholarly visibility"
    assert dashboard.recommended_works[0].explanation.model_scores[0].name == "Hashing baseline pgvector"
