from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from scripts.enrich_semantic_scholar import _doi_id


def enrich(input_path: Path, output_path: Path, limit: int = 200, delay: float = 0.2, session: requests.Session | None = None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Use a separate output file to preserve the OpenAlex source data")
    if limit < 1:
        raise ValueError("limit must be positive")
    http = session or requests.Session()
    http.headers.update({"User-Agent": f"academic-visibility-research/0.1 (mailto:{settings.openalex_mailto})"})
    records = []
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    matched = 0
    attempted = 0
    for work in records:
        doi = _doi_id(work)
        if not doi or attempted >= limit:
            continue
        attempted += 1
        url = f"https://api.crossref.org/works/{quote(doi.removeprefix('DOI:'), safe='')}"
        for attempt in range(4):
            response = http.get(url, timeout=30)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            if attempt == 3:
                response.raise_for_status()
            time.sleep(min(float(response.headers.get("Retry-After", 2 ** attempt)), 60))
        if response.status_code == 404:
            continue
        response.raise_for_status()
        item = response.json().get("message") or {}
        if item.get("DOI", "").lower() != doi.removeprefix("DOI:").lower():
            continue
        work["crossref_activity"] = {
            "doi": item["DOI"],
            "cited_by_count": item.get("is-referenced-by-count"),
            "funders": [funder.get("name") for funder in item.get("funder") or [] if funder.get("name")],
            "awards": [award for funder in item.get("funder") or [] for award in funder.get("award") or []],
            "license_urls": [license_item.get("URL") for license_item in item.get("license") or [] if license_item.get("URL")],
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "source": "Crossref REST API",
        }
        matched += 1
        if delay and attempted < limit:
            time.sleep(delay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for work in records:
            target.write(json.dumps(work, ensure_ascii=False) + "\n")
    return {"input_works": len(records), "doi_lookups": attempted, "matched": matched, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach DOI-matched Crossref citation/funding/license activity with provenance.")
    parser.add_argument("--input", default="data/openalex_temporal_dataset.jsonl")
    parser.add_argument("--output", default="data/openalex_crossref_enriched.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(enrich(Path(args.input), Path(args.output), args.limit, args.delay), indent=2))


if __name__ == "__main__":
    main()
