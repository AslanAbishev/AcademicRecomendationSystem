from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.openalex import OpenAlexClient
from app.config import settings


def build_dataset(query: str, output_path: Path, per_page: int) -> None:
    client = OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    works = client.search_works(query, per_page=per_page)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for work in works:
            authors = [
                {
                    "id": (authorship.get("author") or {}).get("id"),
                    "name": (authorship.get("author") or {}).get("display_name"),
                    "countries": authorship.get("countries") or [],
                    "institutions": [
                        institution.get("display_name")
                        for institution in authorship.get("institutions", [])
                        if institution.get("display_name")
                    ],
                }
                for authorship in work.get("authorships", [])
            ]
            source = ((work.get("primary_location") or {}).get("source") or {})
            record = {
                "id": work.get("id"),
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count", 0),
                "venue": source.get("display_name"),
                "source_type": source.get("type"),
                "topics": [topic.get("display_name") for topic in work.get("topics", []) if topic.get("display_name")],
                "authors": authors,
                "doi": work.get("doi"),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real OpenAlex JSONL dataset slice.")
    parser.add_argument("--query", default="computer science artificial intelligence cybersecurity")
    parser.add_argument("--output", default="data/openalex_dataset.jsonl")
    parser.add_argument("--per-page", type=int, default=200)
    args = parser.parse_args()

    build_dataset(args.query, Path(args.output), args.per_page)


if __name__ == "__main__":
    main()
