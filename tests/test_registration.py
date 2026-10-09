from pathlib import Path
from tempfile import TemporaryDirectory

from app.schemas import AccountRegistrationRequest, ResearcherRegistrationRequest, UserLoginRequest
from app.services.account_service import AccountService
from app.services.registration_service import RegistrationService


class FakeOpenAlexClient:
    def get_author_by_orcid(self, orcid: str):
        return {
            "id": "https://openalex.org/A900",
            "display_name": "Dana Researcher",
            "works_count": 14,
            "cited_by_count": 88,
            "summary_stats": {"h_index": 5, "i10_index": 4},
            "last_known_institutions": [{"display_name": "OpenAlex University", "country_code": "KZ"}],
            "topics": [
                {"display_name": "Recommender Systems", "count": 8},
                {"display_name": "Digital Libraries", "count": 5},
            ],
        }

    def get_author_works(self, author_id: str, per_page: int = 25):
        return [
            {
                "primary_location": {"source": {"type": "conference"}},
                "topics": [{"display_name": "Scholarly Discovery"}],
            },
            {
                "primary_location": {"source": {"type": "journal"}},
                "topics": [{"display_name": "Research Analytics"}],
            },
            {"primary_location": {"source": {"type": "conference"}}},
        ]


def test_registration_service_derives_domains_and_persists_profile():
    payload = ResearcherRegistrationRequest(
        full_name="Dana Researcher",
        email="dana@example.com",
        orcid_id="0000-0000-0000-0000",
        affiliation="Test Lab",
        position_title="Postdoctoral researcher",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        research_keywords=["recommender systems", "scientometrics", "openalex"],
        thematic_clusters=["digital libraries", "scholarly communication"],
        methods=["GraphSAGE", "semantic search"],
        publication_goals=["increase visibility", "publish in information retrieval venues"],
        bio="Works on recommendation, scholarly discovery, and research analytics.",
    )

    with TemporaryDirectory() as tmp_dir:
        service = RegistrationService(Path(tmp_dir) / "profiles.json", client=FakeOpenAlexClient())
        profile = service.register(payload)
        all_profiles = service.list_profiles()

    assert profile.domain_candidates
    assert "Recommender Systems" in [item.name for item in profile.domain_candidates]
    assert "Computer Science" in profile.interest_embedding_text
    assert profile.derived_metrics["h_index"] == 5
    assert profile.derived_metrics["conference_works_count"] == 2
    assert "openalex_orcid" in profile.profile_data_sources
    assert len(all_profiles) == 1


def test_account_service_registers_user_and_authenticates_with_saved_profile():
    payload = AccountRegistrationRequest(
        full_name="Dana Researcher",
        email="Dana@Example.com",
        password="secure-pass-123",
        orcid_id="0000-0000-0000-0000",
        affiliation="Test Lab",
        position_title="Postdoctoral researcher",
        academic_level="ECR",
        academic_field="Computer Science",
        country="Kazakhstan",
        research_keywords=["recommender systems"],
    )

    with TemporaryDirectory() as tmp_dir:
        registration_service = RegistrationService(Path(tmp_dir) / "profiles.json", client=FakeOpenAlexClient())
        account_service = AccountService(Path(tmp_dir) / "users.json")
        created = account_service.register(payload, registration_service)
        logged_in = account_service.login(
            UserLoginRequest(email="dana@example.com", password="secure-pass-123"),
            registration_service,
        )
        current_user = account_service.get_user_by_token(logged_in.access_token)

        users_raw = (Path(tmp_dir) / "users.json").read_text(encoding="utf-8")

    assert created.user.email == "dana@example.com"
    assert logged_in.profile.profile_id == created.profile.profile_id
    assert current_user.profile_id == created.profile.profile_id
    assert "secure-pass-123" not in users_raw


def test_registration_can_be_minimal_when_orcid_enriches_profile():
    payload = AccountRegistrationRequest(
        full_name="Dana Researcher",
        email="dana@example.com",
        password="secure-pass-123",
        orcid_id="0000-0000-0000-0000",
        publication_goals=["increase discoverability"],
    )

    with TemporaryDirectory() as tmp_dir:
        service = RegistrationService(Path(tmp_dir) / "profiles.json", client=FakeOpenAlexClient())
        profile = service.register(payload)

    assert profile.affiliation == "OpenAlex University"
    assert profile.country == "KZ"
    assert profile.academic_field == "Recommender Systems"
    assert profile.research_keywords[:2] == ["Recommender Systems", "Digital Libraries"]
    assert profile.derived_metrics["works_count"] == 14
    assert profile.profile_completeness > 0.6
