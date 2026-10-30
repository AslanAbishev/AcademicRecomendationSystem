from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from scripts.evaluate_collaborators import evaluate
from scripts.train_graph_text_ranker import Period, _digest, _pair_labels, _rank, _read_works


def evaluate_holdout(model_path: Path, training_report_path: Path, frozen_dir: Path, output_path: Path) -> dict:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    train_path = frozen_dir / "train_asof_2025.jsonl"
    labels_path = frozen_dir / "test_new_coauthors_2026.jsonl"
    if model.get("model_name") != "TemporalGraphTextRanker-v2" or training.get("model_name") != model["model_name"]:
        raise ValueError("Expected the locked v2 graph-text model and its training report")
    if _digest(model_path) != manifest["model_sha256"]:
        raise ValueError("The model changed after holdout collection")
    if _digest(train_path) != manifest["train_snapshot"]["sha256"] or _digest(labels_path) != manifest["test_labels"]["sha256"]:
        raise ValueError("The frozen 2026 evaluation inputs changed")
    if manifest["publication_through"] != "2026-09-30" or manifest["holdout_new_coauthor_pairs"] < 30:
        raise ValueError("The locked partial-year holdout has insufficient support")
    k = int(training["k"])
    seed = int(training["seed"])
    max_authors = int(training["candidate_authors_validation"])
    max_queries = int(training["validation_queries"])
    period = Period(_read_works(train_path), max_authors, seed, use_abstracts=True)
    labels = period.labels(_pair_labels(labels_path, expected_year=2026))
    queries = period.queries(labels, max_queries, seed)
    support = sum(len(labels[author]) for author in queries)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "locked v2 artifact; 2020-2025 publication-year features; new partial-2026 links",
        "model_sha256": _digest(model_path),
        "training_report": str(training_report_path),
        "holdout_manifest": str(frozen_dir / "manifest.json"),
        "publication_through": manifest["publication_through"],
        "candidate_authors": len(period.authors),
        "evaluable_queries": len(queries),
        "evaluable_directed_links": support,
        "all_holdout_undirected_links": manifest["holdout_new_coauthor_pairs"],
        "minimum_support": "at least 30 undirected links in the holdout and 20 evaluable query authors",
        "limitations": [
            "2026 ends on September 30 and cannot be compared as a full year.",
            "The author-seeded holdout conditions on active 2023 researchers.",
            "Evaluation queries have at least one observed positive in the sampled candidate pool.",
            "This is a publication-year proxy, not an archival OpenAlex snapshot or causal promotion study.",
        ],
    }
    if len(queries) < 20:
        report["status"] = "insufficient_query_support"
    else:
        graph_text = _rank(period, labels, queries, model, None, k,
                           float(model["fairness_bonus"]), semantic_slots=int(model["semantic_slots"]))
        graph_baseline = _rank(period, labels, queries, None, None, k, 0.0)
        text_baselines = evaluate(train_path, labels_path, k=k, max_authors=max_authors,
                                  max_queries=max_queries, seed=seed)["models"]
        report["status"] = "evaluated_once"
        report["metrics"] = {
            "graph_text_v2": graph_text,
            "graph_adamic_adar": graph_baseline,
            "tfidf": text_baselines["tfidf"],
            "hybrid": text_baselines["hybrid"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked v2 model once on the partial-2026 holdout.")
    parser.add_argument("--model", type=Path, default=Path("data/models/temporal_graph_text_v2.json"))
    parser.add_argument("--training-report", type=Path, default=Path("data/evaluation/temporal_graph_text_v2.json"))
    parser.add_argument("--frozen-dir", type=Path, default=Path("data/multifield/holdout_2026/frozen"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/holdout_2026_v2.json"))
    args = parser.parse_args()
    result = evaluate_holdout(args.model, args.training_report, args.frozen_dir, args.output)
    print(json.dumps({key: result.get(key) for key in ("status", "candidate_authors", "evaluable_queries",
                                                       "evaluable_directed_links", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
