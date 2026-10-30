from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.openalex import OpenAlexClient
from app.config import settings
from scripts.collect_multifield_openalex import DEFAULT_FIELDS


AUTHOR_ID = re.compile(r"^A\d+$")


def _short_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def select_seeds(source: Path, per_field: int = 5, train_through: int = 2023, excluded_authors: set[str] | None = None) -> dict[str, list[str]]:
    if per_field < 1:
        raise ValueError("per_field must be positive")
    excluded = excluded_authors or set()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with source.open(encoding="utf-8") as records:
        for line in records:
            if not line.strip():
                continue
            work = json.loads(line)
            year = int(work.get("publication_year") or 0)
            if year < 2020 or year > train_through:
                continue
            field_id = _short_id(((work.get("primary_topic") or {}).get("field") or {}).get("id"))
            field = next((slug for slug, (known_id, _) in DEFAULT_FIELDS.items() if str(known_id) == field_id), None)
            if not field:
                continue
            authors = {
                _short_id((item.get("author") or {}).get("id"))
                for item in work.get("authorships") or []
            }
            counts[field].update(author for author in authors if AUTHOR_ID.fullmatch(author) and author not in excluded)
    return {
        field: [author for author, count in sorted(counts[field].items(), key=lambda item: (-item[1], item[0])) if count >= 2][:per_field]
        for field in DEFAULT_FIELDS
    }


def expand(source: Path, output_dir: Path, per_field: int = 5, per_author_year: int = 25, client: OpenAlexClient | None = None) -> dict:
    if per_field < 1 or per_author_year < 1:
        raise ValueError("per_field and per_author_year must be positive")
    source_client = client or OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    excluded = set()
    identity_review = output_dir.parent / "identity_review_asof_2023.jsonl"
    if identity_review.exists():
        with identity_review.open(encoding="utf-8") as review:
            for line in review:
                item = json.loads(line)
                excluded.update(item.get("author_ids") or [])
                if item.get("author_id"):
                    excluded.add(item["author_id"])
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("per_field", per_field) != per_field or previous.get("per_author_year", per_author_year) != per_author_year:
            raise ValueError("Use the original cohort settings or a new output directory")
        seeds = previous["seed_authors_by_field"]
    else:
        seeds = select_seeds(source, per_field, excluded_authors=excluded)
    slices = []
    for field, authors in seeds.items():
        for author_id in authors:
            for year in range(2020, 2026):
                target = output_dir / f"{field}_{author_id}_{year}.jsonl"
                done = target.with_suffix(".done")
                if not done.exists():
                    seen: set[str] = set()
                    if target.exists():
                        with target.open(encoding="utf-8") as existing:
                            seen.update(json.loads(line).get("id") for line in existing if line.strip())
                    with target.open("a", encoding="utf-8") as output:
                        for work in source_client.iter_works(
                            "", from_year=year, to_year=year, max_records=per_author_year,
                            extra_filters=[f"authorships.author.id:https://openalex.org/{author_id}"],
                        ):
                            work_id = work.get("id")
                            if not work_id or work_id in seen or int(work.get("publication_year") or 0) != year:
                                continue
                            seen.add(work_id)
                            output.write(json.dumps(work, ensure_ascii=False) + "\n")
                    done.touch()
                with target.open(encoding="utf-8") as collected:
                    count = sum(bool(line.strip()) for line in collected)
                slices.append({"field": field, "seed_author_id": author_id, "year": year, "works": count, "file": str(target)})
        print(f"{field}: {len(authors)} train-period seed authors processed", flush=True)
    manifest = {
        "source": "OpenAlex author-linked works; seed authors chosen from publications through 2023 only",
        "selection": "most frequently observed train-period authors per primary field; longitudinal enrichment is not a representative random sample",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "per_field": per_field,
        "per_author_year": per_author_year,
        "seed_authors_by_field": seeds,
        "slices": slices,
        "works_across_slices_before_deduplication": sum(item["works"] for item in slices),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect longitudinal works for train-period authors in each primary field.")
    parser.add_argument("--input", default="data/openalex_multifield_dataset.jsonl")
    parser.add_argument("--output-dir", default="data/multifield/longitudinal")
    parser.add_argument("--per-field", type=int, default=5)
    parser.add_argument("--per-author-year", type=int, default=25)
    args = parser.parse_args()
    result = expand(Path(args.input), Path(args.output_dir), args.per_field, args.per_author_year)
    print(json.dumps({"seed_authors_by_field": {field: len(authors) for field, authors in result["seed_authors_by_field"].items()}, "works_across_slices_before_deduplication": result["works_across_slices_before_deduplication"]}, indent=2))


if __name__ == "__main__":
    main()
