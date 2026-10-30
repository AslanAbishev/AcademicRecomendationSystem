from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re


AUTHOR_ID = re.compile(r"^A\d+$")


def _short_id(value: str | None) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def _pairs(work: dict) -> list[tuple[str, str]]:
    authors = sorted({
        _short_id((item.get("author") or {}).get("id"))
        for item in work.get("authorships") or []
        if AUTHOR_ID.fullmatch(_short_id((item.get("author") or {}).get("id")))
    })
    if len(authors) > 20:
        return []
    return [(left, right) for index, left in enumerate(authors) for right in authors[index + 1:]]


def _snapshot_work(work: dict) -> dict:
    authorships = []
    for item in work.get("authorships") or []:
        author_id = (item.get("author") or {}).get("id")
        if not AUTHOR_ID.fullmatch(_short_id(author_id)):
            continue
        authorships.append({
            "author": {"id": author_id},
            "raw_author_name": item.get("raw_author_name"),
            "raw_affiliation_strings": item.get("raw_affiliation_strings") or [],
        })
    return {
        "id": work["id"],
        "title": work.get("title") or work.get("display_name"),
        "abstract_inverted_index": work.get("abstract_inverted_index"),
        "publication_year": work["publication_year"],
        "publication_date": work.get("publication_date"),
        "authorships": authorships,
    }


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def freeze(input_path: Path, output_dir: Path, train_through: int = 2023, validation_year: int = 2024, test_year: int = 2025) -> dict:
    if not train_through < validation_year < test_year:
        raise ValueError("Expected train_through < validation_year < test_year")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    by_year: dict[int, list[dict]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            work = json.loads(line)
            year = int(work.get("publication_year") or 0)
            if year <= 0:
                raise ValueError("Every model-ready work must have a publication year")
            by_year[year].append(work)

    train = [work for year, works in sorted(by_year.items()) if year <= train_through for work in works]
    validation = by_year.get(validation_year, [])
    test = [work for year, works in sorted(by_year.items()) if year >= test_year for work in works]
    if not train or not validation or not test:
        raise ValueError("Training, validation, and test periods must all contain works")
    if any(validation_year < year < test_year for year in by_year):
        raise ValueError("No publication years may be skipped between validation and test")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"train_asof_{train_through}.jsonl"
    with train_path.open("w", encoding="utf-8") as target:
        for work in train:
            target.write(json.dumps(_snapshot_work(work), ensure_ascii=False) + "\n")

    train_authors = {
        _short_id((item.get("author") or {}).get("id"))
        for work in train for item in work.get("authorships") or []
    }
    known = {pair for work in train for pair in _pairs(work)}

    def write_labels(works: list[dict], path: Path, prior: set[tuple[str, str]]) -> set[tuple[str, str]]:
        observed: set[tuple[str, str]] = set()
        with path.open("w", encoding="utf-8") as target:
            for work in sorted(works, key=lambda item: (item.get("publication_date") or "", item["id"])):
                field = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name") or "Unknown field"
                for pair in _pairs(work):
                    if pair in prior or pair in observed or not set(pair) <= train_authors:
                        continue
                    observed.add(pair)
                    target.write(json.dumps({
                        "author_a": pair[0], "author_b": pair[1],
                        "publication_year": work["publication_year"],
                        "evidence_work_id": work["id"], "evaluation_field": field,
                    }, ensure_ascii=False) + "\n")
        return observed

    validation_path = output_dir / "validation_new_coauthors.jsonl"
    test_path = output_dir / "test_new_coauthors.jsonl"
    validation_pairs = write_labels(validation, validation_path, known)
    test_pairs = write_labels(test, test_path, known | validation_pairs)
    report = {
        "source": str(input_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "train_through": train_through,
        "validation_year": validation_year,
        "test_year": test_year,
        "train_works": len(train),
        "validation_works": len(validation),
        "test_works": len(test),
        "train_authors": len(train_authors),
        "known_train_pairs": len(known),
        "validation_new_pairs": len(validation_pairs),
        "test_new_pairs": len(test_pairs),
        "artifacts": {
            "train_snapshot": {"path": str(train_path), "sha256": _digest(train_path)},
            "validation_labels": {"path": str(validation_path), "sha256": _digest(validation_path)},
            "test_labels": {"path": str(test_path), "sha256": _digest(test_path)},
        },
        "limitations": [
            "OpenAlex metadata was downloaded later; this is a publication-year proxy, not a historical API snapshot.",
            "The train snapshot strips present-day citation totals, current author ORCIDs, and current topic assignments.",
            "Future works provide labels only and must not be used to fit text, graph, or normalization features.",
            "Missing future collaboration is unlabeled, not confirmed negative evidence.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze train-period features and separate future collaboration labels.")
    parser.add_argument("--input", default="data/openalex_multifield_dataset_model_ready.jsonl")
    parser.add_argument("--output-dir", default="data/multifield/frozen")
    parser.add_argument("--train-through", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    args = parser.parse_args()
    result = freeze(Path(args.input), Path(args.output_dir), args.train_through, args.validation_year, args.test_year)
    print(json.dumps({key: result[key] for key in ("train_works", "train_authors", "validation_new_pairs", "test_new_pairs", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()
