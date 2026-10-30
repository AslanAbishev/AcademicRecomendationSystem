from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time

import requests


ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,year,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf"


def _doi_id(work: dict) -> str | None:
    doi = work.get("doi") or (work.get("ids") or {}).get("doi")
    if not doi:
        return None
    normalized = str(doi).strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return f"DOI:{normalized}" if normalized.startswith("10.") else None


def enrich(input_path: Path, output_path: Path, limit: int = 500, batch_size: int = 50, delay: float = 1.0, session: requests.Session | None = None) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Use a separate output file to preserve the OpenAlex source data")
    if limit < 1 or not 1 <= batch_size <= 500:
        raise ValueError("limit must be positive and batch_size must be 1..500")
    http = session or requests.Session()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    records = []
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    doi_records = [(index, doi) for index, work in enumerate(records) if (doi := _doi_id(work))]
    matched = 0
    attempted = 0
    for offset in range(0, min(limit, len(doi_records)), batch_size):
        batch = doi_records[offset:offset + batch_size]
        ids = [doi for _, doi in batch]
        for attempt in range(5):
            response = http.post(ENDPOINT, params={"fields": FIELDS}, json={"ids": ids}, headers=headers, timeout=45)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if attempt == 4:
                response.raise_for_status()
            time.sleep(min(float(response.headers.get("Retry-After", 2 ** attempt)), 60))
        response.raise_for_status()
        values = response.json()
        if len(values) != len(batch):
            raise ValueError("Semantic Scholar batch response length differs from request length")
        for (index, doi), paper in zip(batch, values, strict=True):
            attempted += 1
            if not paper:
                continue
            records[index]["semantic_scholar_activity"] = {
                "paper_id": paper.get("paperId"),
                "doi_lookup": doi,
                "citation_count": paper.get("citationCount"),
                "influential_citation_count": paper.get("influentialCitationCount"),
                "is_open_access": paper.get("isOpenAccess"),
                "open_access_pdf_url": (paper.get("openAccessPdf") or {}).get("url"),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "source": "Semantic Scholar Academic Graph API",
            }
            matched += 1
        if delay and offset + batch_size < min(limit, len(doi_records)):
            time.sleep(delay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for work in records:
            target.write(json.dumps(work, ensure_ascii=False) + "\n")
    return {"input_works": len(records), "doi_lookups": attempted, "matched": matched, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach DOI-matched Semantic Scholar activity with explicit provenance.")
    parser.add_argument("--input", default="data/openalex_temporal_dataset.jsonl")
    parser.add_argument("--output", default="data/openalex_semantic_scholar_enriched.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(enrich(Path(args.input), Path(args.output), args.limit, args.batch_size, args.delay), indent=2))


if __name__ == "__main__":
    main()
