from app.services.realtime_recommender import ECRRecommenderService


class FakeOpenAlexClient:
    def search_authors(self, query: str, per_page: int = 8):
        if "Recommender Systems" in query:
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


def test_opportunities_are_real_sources_not_fake_catalog_entries():
    service = ECRRecommenderService(client=FakeOpenAlexClient())
    dashboard = service.get_dashboard("A123", top_k=3)

    assert dashboard.opportunities[0].title == "Information Retrieval"
    assert dashboard.opportunities[0].opportunity_type == "journal"
