from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    ids: set[str] = set()
    authors: set[str] = set()
    years: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            work = json.loads(line)
            counts["records"] += 1
            work_id = work.get("id")
            if not work_id:
                counts["missing_id"] += 1
            elif work_id in ids:
                counts["duplicate_id"] += 1
            else:
                ids.add(work_id)
            for field in ("title", "doi", "abstract_inverted_index", "topics", "authorships"):
                if work.get(field):
                    counts[f"with_{field}"] += 1
            try:
                years[int(work["publication_year"])] += 1
            except (KeyError, TypeError, ValueError):
                counts["missing_year"] += 1
            for item in work.get("authorships") or []:
                author_id = (item.get("author") or {}).get("id")
                if author_id:
                    authors.add(author_id)
                if item.get("institutions"):
                    counts["authorships_with_institution"] += 1
                counts["authorships"] += 1
    return {
        "source": str(path),
        "records": counts["records"],
        "unique_works": len(ids),
        "unique_authors": len(authors),
        "years": dict(sorted(years.items())),
        "missing_id": counts["missing_id"],
        "duplicate_id": counts["duplicate_id"],
        "missing_year": counts["missing_year"],
        "with_title": counts["with_title"],
        "with_doi": counts["with_doi"],
        "with_abstract": counts["with_abstract_inverted_index"],
        "with_topics": counts["with_topics"],
        "with_authorships": counts["with_authorships"],
        "authorships_with_institution": counts["authorships_with_institution"],
        "authorships": counts["authorships"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit real OpenAlex JSONL completeness and uniqueness.")
    parser.add_argument("--input", default="data/openalex_author_cohort_trainseed.jsonl")
    parser.add_argument("--output", default="data/evaluation/dataset_quality.json")
    args = parser.parse_args()
    report = inspect(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
