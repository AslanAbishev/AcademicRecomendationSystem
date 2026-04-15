from fastapi.testclient import TestClient

import app.main as main_module
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
                "id": "https://openalex.org/A123",
                "display_name": "Alice Researcher",
                "works_count": 12,
                "cited_by_count": 120,
                "last_known_institutions": [{"display_name": "Test University", "country_code": "KZ"}],
                "topics": [{"display_name": "Recommender Systems", "count": 10}],
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


main_module.service = ECRRecommenderService(client=FakeOpenAlexClient())
client = TestClient(main_module.app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_page_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "Search researcher" in response.text


def test_researcher_search_returns_live_shape():
    response = client.get("/researchers/search?q=alice")
    payload = response.json()

    assert response.status_code == 200
    assert payload[0]["researcher_id"] == "A123"
    assert payload[0]["name"] == "Alice Researcher"


def test_dashboard_returns_real_time_payload_shape():
    response = client.get("/researchers/A123/dashboard")
    payload = response.json()

    assert response.status_code == 200
    assert payload["researcher"]["researcher_id"] == "A123"
    assert payload["recent_works"][0]["title"] == "Fair ranking for scholarly discovery"
    assert len(payload["opportunities"]) >= 1
    assert len(payload["collaborators"]) >= 1
