from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.openalex import OpenAlexClient
from app.config import settings


DEFAULT_FIELDS = {
    "computer_science": (17, "Computer Science"),
    "medicine": (27, "Medicine"),
    "physics": (31, "Physics and Astronomy"),
    "engineering": (22, "Engineering"),
    "social_sciences": (33, "Social Sciences"),
    "biological_sciences": (11, "Agricultural and Biological Sciences"),
}


def _short_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def collect(
    output_dir: Path,
    fields: dict[str, tuple[int, str]] | None = None,
    years: range = range(2020, 2026),
    works_per_field_year: int = 300,
    client: OpenAlexClient | None = None,
) -> dict:
    if works_per_field_year < 1 or not years or fields == {}:
        raise ValueError("Choose fields, years, and a positive number of works per field-year")
    selected = fields if fields is not None else DEFAULT_FIELDS
    source = client or OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = []
    for slug, (field_id, field_name) in selected.items():
        for year in years:
            target = output_dir / f"{slug}_{year}.jsonl"
            existing_ids: set[str] = set()
            if target.exists():
                with target.open(encoding="utf-8") as saved:
                    for line in saved:
                        if line.strip():
                            work_id = _short_id(json.loads(line).get("id"))
                            if work_id:
                                existing_ids.add(work_id)
            added = 0
            if len(existing_ids) < works_per_field_year:
                with target.open("a", encoding="utf-8") as output:
                    for work in source.iter_works(
                        "", from_year=year, to_year=year,
                        max_records=works_per_field_year + len(existing_ids),
                        extra_filters=[f"primary_topic.field.id:{field_id}"],
                    ):
                        work_id = _short_id(work.get("id"))
                        primary_field = (work.get("primary_topic") or {}).get("field") or {}
                        if (
                            not work_id or work_id in existing_ids
                            or int(work.get("publication_year") or 0) != year
                            or _short_id(primary_field.get("id")) != str(field_id)
                        ):
                            continue
                        existing_ids.add(work_id)
                        output.write(json.dumps(work, ensure_ascii=False) + "\n")
                        added += 1
                        if len(existing_ids) >= works_per_field_year:
                            break
                        if added % 100 == 0:
                            output.flush()
            bucket = {
                "field": field_name, "field_id": field_id, "year": year,
                "file": str(target), "works": len(existing_ids), "added": added,
                "target_works": works_per_field_year,
            }
            buckets.append(bucket)
            print(f"{slug} {year}: {len(existing_ids)}/{works_per_field_year} works", flush=True)

    report = {
        "source": "OpenAlex Works API",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "selection": "primary_topic.field.id AND publication year; recent-first API order within year",
        "buckets": buckets,
        "works_across_buckets": sum(item["works"] for item in buckets),
        "all_buckets_complete": all(item["works"] >= works_per_field_year for item in buckets),
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real OpenAlex works across six primary research fields and years.")
    parser.add_argument("--output-dir", default="data/multifield")
    parser.add_argument("--fields", nargs="+", choices=sorted(DEFAULT_FIELDS), default=list(DEFAULT_FIELDS))
    parser.add_argument("--from-year", type=int, default=2020)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--works-per-field-year", type=int, default=300)
    args = parser.parse_args()
    if args.from_year > args.to_year:
        parser.error("from-year must not exceed to-year")
    fields = {name: DEFAULT_FIELDS[name] for name in args.fields}
    report = collect(Path(args.output_dir), fields, range(args.from_year, args.to_year + 1), args.works_per_field_year)
    print(json.dumps({"works_across_buckets": report["works_across_buckets"], "all_buckets_complete": report["all_buckets_complete"]}, indent=2))


if __name__ == "__main__":
    main()
