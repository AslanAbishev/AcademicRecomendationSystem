"""Isolated UI smoke-test server. Fixtures never enter the real dataset.

Run with python -m tests.ui_preview. All account and progress writes use a
temporary directory. The normal app and Docker Compose never import this file.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from tests.test_api import FakeOpenAlexClient
import app.main as main
from app.services.account_service import AccountService
from app.services.progress_service import PromotionProgressService
from app.services.realtime_recommender import ECRRecommenderService
from app.services.registration_service import RegistrationService


class PreviewClient(FakeOpenAlexClient):
    def get_author_works(self, author_id: str, per_page: int = 12):
        # Repeated, explicitly labeled fixtures exercise publication pagination.
        work = super().get_author_works(author_id, per_page)[0]
        works = []
        for index in range(9):
            item = deepcopy(work)
            item["id"] = f"https://openalex.org/W{index + 1}"
            item["title"] = f"UI test fixture {index + 1}: {work['title']}"
            works.append(item)
        return works[:per_page]


class PreviewRecommender(ECRRecommenderService):
    def _recommend_scientific_works(self, researcher_text, embedding_model, top_k, exclude_work_ids=None):
        return [], f"{embedding_model}: paper embeddings are disabled in this isolated UI test."


if __name__ == "__main__":
    with TemporaryDirectory(prefix="ecr-ui-test-") as directory:
        root = Path(directory)
        client = PreviewClient()
        main.service = PreviewRecommender(client=client)
        main.registration_service = RegistrationService(root / "profiles.json", client=client)
        main.account_service = AccountService(root / "users.json")
        main.progress_service = PromotionProgressService(root / "snapshots.json", root / "actions.json")
        uvicorn.run(main.app, host="0.0.0.0", port=8001)
