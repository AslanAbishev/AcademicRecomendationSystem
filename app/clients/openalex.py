from __future__ import annotations

from typing import Any

import requests


class OpenAlexClient:
    def __init__(self, mailto: str = "researcher@example.com", timeout: int = 30, api_key: str | None = None) -> None:
        self.mailto = mailto
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ecr-visibility-mvp/0.2"})

    def search_authors(
        self,
        query: str | None = None,
        per_page: int = 8,
        extra_filters: list[str] | None = None,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "per-page": per_page,
            "mailto": self.mailto,
        }
        if query:
            params["search"] = query
        if extra_filters:
            params["filter"] = ",".join(extra_filters)
        if sort:
            params["sort"] = sort
        payload = self._get(
            "/authors",
            params=params,
        )
        return payload.get("results", [])

    def get_author(self, author_id: str) -> dict[str, Any]:
        normalized = author_id if author_id.startswith("A") else author_id.split("/")[-1]
        return self._get(f"/authors/{normalized}", params={"mailto": self.mailto})

    def get_author_by_orcid(self, orcid: str) -> dict[str, Any]:
        normalized = orcid.strip()
        if not normalized:
            raise ValueError("ORCID must not be empty")
        if normalized.startswith("https://orcid.org/"):
            external_id = normalized
        else:
            external_id = f"orcid:{normalized}"
        return self._get(f"/authors/{external_id}", params={"mailto": self.mailto})

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

    def search_works(
        self,
        query: str,
        per_page: int = 12,
        extra_filters: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filters = ["from_publication_date:2020-01-01", *(extra_filters or [])]
        payload = self._get(
            "/works",
            params={
                "search": query,
                "filter": ",".join(filters),
                "sort": "cited_by_count:desc",
                "per-page": per_page,
                "mailto": self.mailto,
            },
        )
        return payload.get("results", [])

    def iter_works(
        self,
        query: str,
        per_page: int = 200,
        max_records: int = 1000,
        extra_filters: list[str] | None = None,
        from_year: int = 2020,
        to_year: int | None = None,
        sort: str = "publication_year:desc",
    ):
        filters = [f"from_publication_date:{from_year}-01-01", *(extra_filters or [])]
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")

        cursor = "*"
        yielded = 0
        while yielded < max_records:
            payload = self._get(
                "/works",
                params={
                    "search": query,
                    "filter": ",".join(filters),
                    "sort": sort,
                    "per-page": min(per_page, max_records - yielded),
                    "cursor": cursor,
                    "mailto": self.mailto,
                },
            )
            results = payload.get("results", [])
            if not results:
                break
            for item in results:
                yield item
                yielded += 1
                if yielded >= max_records:
                    break
            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break

    def search_institutions(self, query: str, per_page: int = 5) -> list[dict[str, Any]]:
        payload = self._get(
            "/institutions",
            params={
                "search": query,
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
        if self.api_key:
            params = {**params, "api_key": self.api_key}
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
