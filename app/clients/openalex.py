from __future__ import annotations

from typing import Any

import requests


class OpenAlexClient:
    def __init__(self, mailto: str = "researcher@example.com", timeout: int = 30) -> None:
        self.mailto = mailto
        self.timeout = timeout
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ecr-visibility-mvp/0.2"})

    def search_authors(self, query: str, per_page: int = 8) -> list[dict[str, Any]]:
        payload = self._get(
            "/authors",
            params={
                "search": query,
                "per-page": per_page,
                "mailto": self.mailto,
            },
        )
        return payload.get("results", [])

    def get_author(self, author_id: str) -> dict[str, Any]:
        normalized = author_id if author_id.startswith("A") else author_id.split("/")[-1]
        return self._get(f"/authors/{normalized}", params={"mailto": self.mailto})

    def get_author_works(self, author_id: str, per_page: int = 12) -> list[dict[str, Any]]:
        normalized = author_id if author_id.startswith("A") else author_id.split("/")[-1]
        payload = self._get(
            "/works",
            params={
                "filter": f"authorships.author.id:https://openalex.org/{normalized}",
                "sort": "publication_year:desc",
                "per-page": per_page,
                "mailto": self.mailto,
            },
        )
        return payload.get("results", [])

    def search_sources(self, query: str, per_page: int = 8) -> list[dict[str, Any]]:
        payload = self._get(
            "/sources",
            params={
                "search": query,
                "per-page": per_page,
                "mailto": self.mailto,
            },
        )
        return payload.get("results", [])

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
