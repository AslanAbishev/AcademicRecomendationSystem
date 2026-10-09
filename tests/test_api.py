from fastapi.testclient import TestClient
from pathlib import Path
from tempfile import TemporaryDirectory

import app.main as main_module
from app.services.account_service import AccountService
from app.services.progress_service import PromotionProgressService
from app.services.registration_service import RegistrationService
from app.services.realtime_recommender import ECRRecommenderService


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
                        "institutions": [{"display_name": "Test University", "country_code": "KZ"}],
                        "countries": ["KZ"],
                    },
                ],
            }
        ]

    def search_institutions(self, query: str, per_page: int = 5):
        return [
            {"id": "https://openalex.org/I123", "display_name": "Test University Research Center"},
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
    assert "Your work." in response.text
    assert 'aria-label="Workspace navigation"' in response.text
    assert 'src="/static/app.js' in response.text


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
    assert {item["name"] for item in payload["collaborators"][0]["explanation"]["model_scores"]} >= {
        "TF-IDF cosine",
        "BM25",
        "LSA/SVD",
        "Semantic hashing",
        "Hybrid ensemble",
    }


def test_dashboard_accepts_collaborator_scope_filter():
    response = client.get("/researchers/A123/dashboard?collaborator_scope=same_country")
    payload = response.json()

    assert response.status_code == 200
    assert payload["collaborator_scope"] == "same_country"
    assert {item["country_code"] for item in payload["collaborators"]} == {"KZ"}


def test_profile_registration_returns_domain_candidates():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(
            Path(tmp_dir) / "registered_profiles.json",
            client=FakeOpenAlexClient(),
        )
        response = client.post(
            "/profiles/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
                "orcid_id": "0000-0000-0000-0000",
                "research_keywords": ["recommender systems", "scholarly discovery", "fairness"],
                "thematic_clusters": ["digital libraries", "scientometrics"],
                "methods": ["SPECTER2", "GraphSAGE"],
                "publication_goals": ["increase discoverability"],
                "bio": "Research on recommendation, ranking, and scholarly communication.",
            },
        )
        payload = response.json()

    assert response.status_code == 200
    assert payload["full_name"] == "Aslan Abishev"
    assert payload["domain_candidates"][0]["name"] == "Recommender Systems"
    assert payload["derived_metrics"]["citation_count"] == 120
    assert "openalex_orcid" in payload["profile_data_sources"]


def test_registered_profile_can_be_loaded_by_id():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(Path(tmp_dir) / "registered_profiles.json")
        created = client.post(
            "/profiles/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
            },
        ).json()
        response = client.get(f"/profiles/registered/{created['profile_id']}")

    assert response.status_code == 200
    assert response.json()["profile_id"] == created["profile_id"]


def test_registered_profile_dashboard_uses_profile_context():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(
            Path(tmp_dir) / "registered_profiles.json",
            client=FakeOpenAlexClient(),
        )
        created = client.post(
            "/profiles/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
                "orcid_id": "0000-0000-0000-0000",
                "research_keywords": ["recommender systems", "scholarly discovery"],
                "bio": "Research on recommendation and scholarly discovery.",
            },
        ).json()
        response = client.get(f"/profiles/registered/{created['profile_id']}/dashboard")

    payload = response.json()
    assert response.status_code == 200
    assert payload["researcher"]["researcher_id"] == "A123"
    assert payload["recent_works"][0]["title"] == "Fair ranking for scholarly discovery"


def test_registered_profile_cabinet_returns_profile_and_recommendations():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(
            Path(tmp_dir) / "registered_profiles.json",
            client=FakeOpenAlexClient(),
        )
        created = client.post(
            "/profiles/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
                "orcid_id": "0000-0000-0000-0000",
                "google_scholar_url": "https://scholar.google.com/citations?user=test",
                "research_keywords": ["recommender systems", "scholarly discovery"],
                "bio": "Research on recommendation and scholarly discovery.",
            },
        ).json()
        response = client.get(f"/profiles/registered/{created['profile_id']}/cabinet")

    payload = response.json()
    assert response.status_code == 200
    assert payload["registered_profile"]["profile_id"] == created["profile_id"]
    assert payload["personalized_dashboard"]["opportunities"][0]["title"] == "Information Retrieval"
    assert payload["connected_profiles"][0]["platform"] == "ORCID"


def test_auth_register_login_and_me_cabinet_flow():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(
            Path(tmp_dir) / "registered_profiles.json",
            client=FakeOpenAlexClient(),
        )
        main_module.account_service = AccountService(Path(tmp_dir) / "users.json")
        main_module.progress_service = PromotionProgressService(
            Path(tmp_dir) / "promotion_snapshots.json",
            Path(tmp_dir) / "promotion_actions.json",
        )

        register_response = client.post(
            "/auth/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "password": "secure-pass-123",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
                "orcid_id": "0000-0000-0000-0000",
                "research_keywords": ["recommender systems", "scholarly discovery"],
                "bio": "Research on recommendation and scholarly discovery.",
            },
        )
        token = register_response.json()["access_token"]

        cabinet_response = client.get(
            "/auth/me/cabinet",
            headers={"Authorization": f"Bearer {token}"},
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "aslan@example.com", "password": "secure-pass-123"},
        )

    assert register_response.status_code == 200
    assert register_response.json()["profile"]["full_name"] == "Aslan Abishev"
    assert cabinet_response.status_code == 200
    assert cabinet_response.json()["cabinet"]["registered_profile"]["email"] == "aslan@example.com"
    assert login_response.status_code == 200
    assert login_response.json()["user"]["email"] == "aslan@example.com"


def test_auth_progress_snapshot_and_action_feedback_flow():
    with TemporaryDirectory() as tmp_dir:
        main_module.registration_service = RegistrationService(
            Path(tmp_dir) / "registered_profiles.json",
            client=FakeOpenAlexClient(),
        )
        main_module.account_service = AccountService(Path(tmp_dir) / "users.json")
        main_module.progress_service = PromotionProgressService(
            Path(tmp_dir) / "promotion_snapshots.json",
            Path(tmp_dir) / "promotion_actions.json",
        )

        register_response = client.post(
            "/auth/register",
            json={
                "full_name": "Aslan Abishev",
                "email": "aslan@example.com",
                "password": "secure-pass-123",
                "affiliation": "Korkyt Ata University",
                "position_title": "PhD student",
                "academic_level": "early career researcher",
                "academic_field": "Computer Science",
                "country": "Kazakhstan",
                "orcid_id": "0000-0000-0000-0000",
                "research_keywords": ["recommender systems", "scholarly discovery"],
            },
        )
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        first_snapshot = client.post("/auth/me/snapshots", headers=headers)
        action_response = client.post(
            "/auth/me/actions",
            headers=headers,
            json={"title": "Connect ORCID to verify publication history", "status": "completed"},
        )
        second_snapshot = client.post("/auth/me/snapshots", headers=headers)
        progress_response = client.get("/auth/me/progress", headers=headers)

    assert first_snapshot.status_code == 200
    assert action_response.status_code == 200
    assert second_snapshot.status_code == 200
    assert progress_response.status_code == 200
    assert progress_response.json()["latest"]["completed_actions"] == 1
    assert progress_response.json()["deltas"]["completed_actions"] == 1


def test_experiment_ml_plan_documents_models_and_metrics():
    response = client.get("/experiments/ml-plan")
    payload = response.json()

    assert response.status_code == 200
    assert payload["dataset_source"] == "OpenAlex"
    assert any("TF-IDF" in model for model in payload["models"])
    assert any("NDCG" in metric for metric in payload["metrics"])
