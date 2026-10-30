from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path

from app.clients.openalex import OpenAlexClient
from app.config import settings


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def select_train_seed_authors(train_path: Path, count: int) -> list[str]:
    if count < 1:
        raise ValueError("Seed author count must be positive")
    recent = Counter()
    with train_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            work = json.loads(line)
            if int(work.get("publication_year") or 0) != 2023:
                continue
            for item in work.get("authorships") or []:
                author = ((item.get("author") or {}).get("id") or "").split("/")[-1]
                if author.startswith("A") and author[1:].isdigit():
                    recent[author] += 1
    if not recent:
        raise ValueError("No 2023 authors found in the frozen training snapshot")
    return [author for author, _ in sorted(recent.items(), key=lambda pair: (-pair[1], pair[0]))[:count]]


def collect(train_path: Path, model_path: Path, output_dir: Path, author_count: int = 60,
            works_per_author: int = 30, cutoff: str = "2026-09-30", client: OpenAlexClient | None = None) -> dict:
    cutoff_date = date.fromisoformat(cutoff)
    if cutoff_date.year != 2026 or works_per_author < 1:
        raise ValueError("Choose a 2026 cutoff and positive per-author work limit")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("model_name") != "TemporalGraphTextRanker-v2":
        raise ValueError("Freeze the v2 model before collecting a prospective holdout")
    if model.get("train_snapshot_sha256") != _digest(train_path):
        raise ValueError("Model and training snapshot do not match")
    seed_authors = select_train_seed_authors(train_path, author_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "source": "OpenAlex API",
        "selection": "highest 2023 work counts among authors in the frozen train snapshot; no 2024-2026 labels used",
        "publication_from": "2026-01-01",
        "publication_through": cutoff,
        "model_sha256_before_collection": _digest(model_path),
        "train_snapshot_sha256": _digest(train_path),
        "seed_authors": seed_authors,
        "works_per_author_limit": works_per_author,
        "status": "collecting",
        "caveats": [
            "A partial 2026 year cannot be compared directly with a complete 2024 or 2025 year.",
            "The intentionally active-author cohort is not representative of all young scientists.",
            "OpenAlex records are current API metadata, not archival publication-date snapshots.",
            "A missing future link is unlabeled, not a confirmed non-collaboration.",
        ],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        locked = ("publication_from", "publication_through", "model_sha256_before_collection",
                  "train_snapshot_sha256", "seed_authors", "works_per_author_limit")
        if any(previous.get(key) != manifest.get(key) for key in locked):
            raise ValueError("Prospective holdout protocol is already frozen with different inputs")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    source = client or OpenAlexClient(mailto=settings.openalex_mailto, api_key=settings.openalex_api_key)
    records: dict[str, dict] = {}
    for author in seed_authors:
        target = output_dir / f"{author}.jsonl"
        done = output_dir / f"{author}.done"
        if not done.exists():
            seen = set()
            with target.open("w", encoding="utf-8") as output:
                for work in source.iter_works(
                    "", from_year=2026, max_records=works_per_author, per_page=min(works_per_author, 100),
                    extra_filters=[f"authorships.author.id:https://openalex.org/{author}",
                                   f"to_publication_date:{cutoff}"],
                ):
                    publication_date = work.get("publication_date") or ""
                    work_id = work.get("id")
                    if not work_id or work_id in seen or not "2026-01-01" <= publication_date <= cutoff:
                        continue
                    seen.add(work_id)
                    output.write(json.dumps(work, ensure_ascii=False) + "\n")
            done.touch()
        with target.open(encoding="utf-8") as source_file:
            for line in source_file:
                if line.strip():
                    work = json.loads(line)
                    records[work["id"]] = work
    works_path = output_dir / "works_2026_partial.jsonl"
    with works_path.open("w", encoding="utf-8") as output:
        for work_id in sorted(records):
            output.write(json.dumps(records[work_id], ensure_ascii=False) + "\n")
    manifest.update({
        "status": "complete",
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "completed_seed_authors": len(seed_authors),
        "unique_works": len(records),
        "works_file": str(works_path),
        "works_sha256": _digest(works_path),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a frozen partial-2026 OpenAlex holdout after model lock.")
    parser.add_argument("--train", type=Path, default=Path("data/multifield/frozen/train_asof_2023.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("data/models/temporal_graph_text_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multifield/holdout_2026"))
    parser.add_argument("--author-count", type=int, default=60)
    parser.add_argument("--works-per-author", type=int, default=30)
    parser.add_argument("--cutoff", default="2026-09-30")
    args = parser.parse_args()
    result = collect(args.train, args.model, args.output_dir,
                     args.author_count, args.works_per_author, args.cutoff)
    print(json.dumps({key: result[key] for key in ("status", "publication_through", "completed_seed_authors",
                                                   "unique_works", "works_file", "works_sha256")}, indent=2))


if __name__ == "__main__":
    main()
