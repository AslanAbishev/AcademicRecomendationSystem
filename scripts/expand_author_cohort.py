from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.openalex import OpenAlexClient
from app.config import settings


def expand(input_path: Path, output_path: Path, max_authors: int = 200, max_works_per_author: int = 100, seed_train_through: int = 2023, client: OpenAlexClient | None = None) -> dict:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Use separate input and output files")
    if max_authors < 1 or max_works_per_author < 1:
        raise ValueError("Author and work limits must be positive")
    source = client or OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    author_frequency: Counter[str] = Counter()
    seen: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    with input_path.open(encoding="utf-8") as input_file, output_path.open(mode, encoding="utf-8") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            work = json.loads(line)
            work_id = work.get("id")
            if work_id and work_id not in seen:
                seen.add(work_id)
                if mode == "w":
                    output_file.write(line.rstrip("\n") + "\n")
            if work.get("publication_year") and int(work["publication_year"]) <= seed_train_through:
                author_frequency.update({
                    (item.get("author") or {}).get("id", "").split("/")[-1]
                    for item in work.get("authorships") or []
                    if (item.get("author") or {}).get("id")
                })
        if mode == "a":
            with output_path.open(encoding="utf-8") as existing:
                seen.update(json.loads(line).get("id") for line in existing if line.strip())
        seeds = [author_id for author_id, count in sorted(author_frequency.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:max_authors]
        added = 0
        for position, author_id in enumerate(seeds, 1):
            for work in source.iter_works(
                "", from_year=2020, to_year=2025,
                max_records=max_works_per_author,
                extra_filters=[f"authorships.author.id:https://openalex.org/{author_id}"],
            ):
                work_id = work.get("id")
                if not work_id or work_id in seen:
                    continue
                seen.add(work_id)
                output_file.write(json.dumps(work, ensure_ascii=False) + "\n")
                added += 1
            if position % 10 == 0:
                output_file.flush()
                print(f"authors_processed={position} works_added={added}", flush=True)
    return {
        "source": "OpenAlex works and author-linked works",
        "input": str(input_path), "output": str(output_path),
        "seed_authors": len(seeds), "added_works": added,
        "seed_train_through": seed_train_through,
        "unique_works": len(seen), "collected_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand real OpenAlex works around repeat authors in a seed corpus.")
    parser.add_argument("--input", default="data/openalex_temporal_dataset.jsonl")
    parser.add_argument("--output", default="data/openalex_author_cohort_trainseed.jsonl")
    parser.add_argument("--max-authors", type=int, default=120)
    parser.add_argument("--max-works-per-author", type=int, default=100)
    parser.add_argument("--seed-train-through", type=int, default=2023)
    args = parser.parse_args()
    print(json.dumps(expand(Path(args.input), Path(args.output), args.max_authors, args.max_works_per_author, args.seed_train_through), indent=2))


if __name__ == "__main__":
    main()
