from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from scripts.freeze_temporal_cohort import _pairs, _short_id, _snapshot_work


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def prepare(model_ready: Path, collection_dir: Path, model_path: Path, output_dir: Path) -> dict:
    collection = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    if collection["status"] != "complete" or collection["model_sha256_before_collection"] != _digest(model_path):
        raise ValueError("The holdout collection must be complete and tied to the frozen model")
    works_2026 = collection_dir / "works_2026_partial.jsonl"
    if collection["works_sha256"] != _digest(works_2026):
        raise ValueError("Collected 2026 works changed after collection")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "train_asof_2025.jsonl"
    known_pairs: set[tuple[str, str]] = set()
    known_authors: set[str] = set()
    years = Counter()
    with snapshot_path.open("w", encoding="utf-8") as output:
        for work in _read(model_ready):
            year = int(work.get("publication_year") or 0)
            if year < 2020 or year > 2025:
                continue
            snapshot = _snapshot_work(work)
            output.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            years[year] += 1
            known_pairs.update(_pairs(snapshot))
            known_authors.update(_short_id((item.get("author") or {}).get("id"))
                                 for item in snapshot["authorships"])
    if not years[2025]:
        raise ValueError("The pre-2026 profile snapshot must include 2025 works")
    labels_path = output_dir / "test_new_coauthors_2026.jsonl"
    observed: set[tuple[str, str]] = set()
    quality = Counter()
    with labels_path.open("w", encoding="utf-8") as output:
        for work in sorted(_read(works_2026), key=lambda item: (item.get("publication_date") or "", item["id"])):
            publication_date = work.get("publication_date") or ""
            if not "2026-01-01" <= publication_date <= collection["publication_through"]:
                raise ValueError("A collected work falls outside the locked holdout date range")
            quality["works"] += 1
            if len(work.get("authorships") or []) > 20:
                quality["large_author_works_excluded"] += 1
                continue
            for left, right in _pairs(work):
                if left not in known_authors or right not in known_authors:
                    quality["pairs_with_new_authors_excluded"] += 1
                    continue
                if (left, right) in known_pairs or (left, right) in observed:
                    continue
                observed.add((left, right))
                output.write(json.dumps({
                    "author_a": left, "author_b": right,
                    "publication_year": 2026,
                    "publication_date": publication_date,
                    "evidence_work_id": work["id"],
                }, ensure_ascii=False) + "\n")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": _digest(model_path),
        "source_model_ready": str(model_ready),
        "source_model_ready_sha256": _digest(model_ready),
        "collection_manifest": str(collection_dir / "manifest.json"),
        "publication_through": collection["publication_through"],
        "train_works_by_year": dict(sorted(years.items())),
        "train_authors": len(known_authors),
        "known_pre_2026_coauthor_pairs": len(known_pairs),
        "holdout_works": quality["works"],
        "holdout_new_coauthor_pairs": len(observed),
        "quality": dict(quality),
        "train_snapshot": {"path": str(snapshot_path), "sha256": _digest(snapshot_path)},
        "test_labels": {"path": str(labels_path), "sha256": _digest(labels_path)},
        "limitations": [
            "The 2026 period ends on the locked cutoff and is not a full-year test.",
            "The author-seeded collection is not representative of all researchers or disciplines.",
            "OpenAlex work metadata was retrieved later than the publication dates.",
            "Only author pairs already observed before 2026 are evaluable; missing outcomes are unlabeled.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze pre-2026 features and new partial-2026 coauthor labels.")
    parser.add_argument("--model-ready", type=Path, default=Path("data/openalex_multifield_dataset_model_ready.jsonl"))
    parser.add_argument("--collection-dir", type=Path, default=Path("data/multifield/holdout_2026"))
    parser.add_argument("--model", type=Path, default=Path("data/models/temporal_graph_text_v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multifield/holdout_2026/frozen"))
    args = parser.parse_args()
    result = prepare(args.model_ready, args.collection_dir, args.model, args.output_dir)
    print(json.dumps({key: result[key] for key in ("publication_through", "holdout_works",
                                                   "holdout_new_coauthor_pairs", "train_authors", "quality")}, indent=2))


if __name__ == "__main__":
    main()
