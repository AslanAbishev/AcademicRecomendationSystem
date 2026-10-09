from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_through: int,
    validation_year: int,
    test_from: int,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {input_path}")
    if not train_through < validation_year < test_from:
        raise ValueError("Expected train_through < validation_year < test_from")

    records: list[dict] = []
    seen_ids: set[str] = set()
    duplicates = 0
    missing_year = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        work_id = str(record.get("id") or record.get("work_id") or "")
        if work_id and work_id in seen_ids:
            duplicates += 1
            continue
        if work_id:
            seen_ids.add(work_id)
        year = _publication_year(record)
        if year is None:
            missing_year += 1
            continue
        record["_publication_year"] = year
        records.append(record)

    buckets = {
        "train": [record for record in records if record["_publication_year"] <= train_through],
        "validation": [record for record in records if record["_publication_year"] == validation_year],
        "test": [record for record in records if record["_publication_year"] >= test_from],
    }
    assigned = sum(len(items) for items in buckets.values())
    unassigned = len(records) - assigned
    if unassigned:
        raise ValueError(
            f"{unassigned} records fall into the gap between validation_year and test_from. "
            "Use consecutive years or adjust the split arguments."
        )
    empty_buckets = [name for name, items in buckets.items() if not items]
    if empty_buckets:
        counts = ", ".join(f"{name}={len(items)}" for name, items in buckets.items())
        raise ValueError(
            "Temporal split is not valid because one or more buckets are empty "
            f"({counts}). Collect works from multiple publication years before training."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in buckets.items():
        target = output_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as file:
            for record in items:
                clean_record = dict(record)
                clean_record.pop("_publication_year", None)
                file.write(json.dumps(clean_record, ensure_ascii=False) + chr(10))

    metadata = {
        "source": str(input_path),
        "records_read": len(records) + duplicates + missing_year,
        "records_written": assigned,
        "duplicates_removed": duplicates,
        "records_without_publication_year": missing_year,
        "split": {
            "train": f"publication_year <= {train_through}",
            "validation": f"publication_year == {validation_year}",
            "test": f"publication_year >= {test_from}",
        },
        "counts": {name: len(items) for name, items in buckets.items()},
        "years": {
            name: sorted({_publication_year(item) for item in items if _publication_year(item) is not None})
            for name, items in buckets.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _publication_year(record: dict) -> int | None:
    value = record.get("publication_year")
    if value is None:
        value = (record.get("biblio") or {}).get("publication_year")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a leakage-aware temporal train/validation/test split for OpenAlex works."
    )
    parser.add_argument("--input", default="data/openalex_cs_ai_cyber_2020_2025_sample.jsonl")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-through", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-from", type=int, default=2025)
    args = parser.parse_args()
    print(
        json.dumps(
            split_dataset(
                Path(args.input),
                Path(args.output_dir),
                args.train_through,
                args.validation_year,
                args.test_from,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
