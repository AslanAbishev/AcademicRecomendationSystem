from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_TOPICS = [
    "academic recommender systems",
    "early career researcher visibility",
    "fairness in scholarly search",
    "scientific collaboration recommendation",
]


def fetch_works(query: str, per_page: int = 25, max_pages: int = 2) -> list[dict]:
    works: list[dict] = []
    cursor = "*"

    for _ in range(max_pages):
        response = requests.get(
            OPENALEX_WORKS_URL,
            params={
                "search": query,
                "per-page": per_page,
                "cursor": cursor,
                "mailto": "researcher@example.com",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        works.extend(payload.get("results", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1)

    return works


def build_datasets(queries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_rows: list[dict] = []
    author_index: dict[str, dict] = {}

    for query in queries:
        works = fetch_works(query)
        for work in works:
            title = work.get("title") or ""
            abstract = work.get("abstract_inverted_index") or {}
            abstract_terms = sorted(
                ((position, token) for token, positions in abstract.items() for position in positions),
                key=lambda item: item[0],
            )
            abstract_text = " ".join(token for _, token in abstract_terms[:400])
            concepts = [item.get("display_name", "") for item in work.get("concepts", [])[:5] if item.get("display_name")]
            work_rows.append(
                {
                    "work_id": work.get("id"),
                    "title": title,
                    "year": work.get("publication_year"),
                    "type": work.get("type"),
                    "cited_by_count": work.get("cited_by_count"),
                    "topics": "|".join(concepts),
                    "abstract_excerpt": abstract_text,
                    "source_query": query,
                }
            )

            for author in work.get("authorships", [])[:8]:
                author_info = author.get("author", {})
                author_id = author_info.get("id")
                if not author_id:
                    continue
                record = author_index.setdefault(
                    author_id,
                    {
                        "author_id": author_id,
                        "name": author_info.get("display_name", "Unknown"),
                        "works_count": 0,
                        "topics": set(),
                        "institutions": set(),
                        "countries": set(),
                    },
                )
                record["works_count"] += 1
                record["topics"].update(concepts)
                for institution in author.get("institutions", []):
                    if institution.get("display_name"):
                        record["institutions"].add(institution["display_name"])
                    if institution.get("country_code"):
                        record["countries"].add(institution["country_code"])

    works_df = pd.DataFrame(work_rows).drop_duplicates(subset=["work_id"]).reset_index(drop=True)
    authors_df = pd.DataFrame(
        [
            {
                "author_id": author_id,
                "name": data["name"],
                "works_count": data["works_count"],
                "topics": "|".join(sorted(item for item in data["topics"] if item)),
                "institutions": "|".join(sorted(data["institutions"])),
                "countries": "|".join(sorted(data["countries"])),
            }
            for author_id, data in author_index.items()
        ]
    ).sort_values(by=["works_count", "name"], ascending=[False, True])

    return authors_df, works_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex seed data for the ECR MVP.")
    parser.add_argument("--query", action="append", dest="queries", help="Search query to fetch from OpenAlex.")
    parser.add_argument("--output-dir", default="data", help="Directory where CSV files will be written.")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_TOPICS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    authors_df, works_df = build_datasets(queries)
    authors_path = output_dir / "openalex_researchers_generated.csv"
    works_path = output_dir / "openalex_works_generated.csv"
    authors_df.to_csv(authors_path, index=False)
    works_df.to_csv(works_path, index=False)

    print(f"Saved {len(authors_df)} authors to {authors_path}")
    print(f"Saved {len(works_df)} works to {works_path}")


if __name__ == "__main__":
    main()
