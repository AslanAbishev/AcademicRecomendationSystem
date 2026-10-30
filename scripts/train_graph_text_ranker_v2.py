from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.train_graph_text_ranker import (
    FEATURES,
    Period,
    _digest,
    _future_edges,
    _pair_labels,
    _rank,
    _read_works,
    _training_pairs,
)


def train_v2(train_path: Path, validation_path: Path, model_path: Path, report_path: Path,
             max_authors: int = 3000, max_queries: int = 100, k: int = 10, seed: int = 42) -> dict:
    works = _read_works(train_path)
    years = {int(work["publication_year"]) for work in works}
    if not years or min(years) < 2020 or max(years) != 2023:
        raise ValueError("Expected only 2020-2023 frozen training works")
    earlier = [work for work in works if int(work["publication_year"]) <= 2022]
    year_2023 = [work for work in works if int(work["publication_year"]) == 2023]
    fit_period = Period(earlier, max_authors, seed, use_abstracts=True)
    fit_labels = fit_period.labels(_future_edges(year_2023))
    fit_queries = fit_period.queries(fit_labels, max_queries, seed)
    x, y = _training_pairs(fit_period, fit_labels, fit_queries, seed)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed).fit(scaler.transform(x), y)

    period = Period(works, max_authors, seed, use_abstracts=True)
    labels = period.labels(_pair_labels(validation_path))
    queries = period.queries(labels, max_queries, seed)
    if not queries:
        raise ValueError("No 2024 validation links in the frozen candidate pool")
    graph_baseline = _rank(period, labels, queries, None, None, k, 0.0)
    variants = {}
    for fairness_bonus in (0.0, 0.5, 1.0):
        for semantic_slots in (0, 1, 2):
            key = f"fairness={fairness_bonus};semantic_slots={semantic_slots}"
            variants[key] = _rank(period, labels, queries, model, scaler, k, fairness_bonus,
                                  semantic_slots=semantic_slots)
    feasible = [key for key, metrics in variants.items()
                if metrics["coverage"] >= graph_baseline["coverage"]
                and metrics["fairness_exposure_gap"] <= graph_baseline["fairness_exposure_gap"]]
    pool = feasible or list(variants)
    selected = max(pool, key=lambda key: (
        variants[key]["ndcg_at_k"], variants[key]["coverage"], -variants[key]["fairness_exposure_gap"]))
    selected_fairness, selected_slots = selected.split(";")
    fairness_bonus = float(selected_fairness.split("=")[1])
    semantic_slots = int(selected_slots.split("=")[1])
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "TemporalGraphTextRanker-v2",
        "train_snapshot": str(train_path),
        "train_snapshot_sha256": _digest(train_path),
        "validation_labels": str(validation_path),
        "validation_labels_sha256": _digest(validation_path),
        "fit_features_through": 2022,
        "fit_labels_year": 2023,
        "validation_features_through": 2023,
        "validation_labels_year": 2024,
        "test_year": None,
        "abstracts_per_author_max": 4,
        "positive_training_examples": int(y.sum()),
        "unlabeled_training_examples": int(len(y) - y.sum()),
        "fit_queries": len(fit_queries),
        "validation_queries": len(queries),
        "candidate_authors_fit": len(fit_period.authors),
        "candidate_authors_validation": len(period.authors),
        "k": k, "seed": seed,
        "selection_constraint": "coverage >= Adamic-Adar and exposure gap <= Adamic-Adar on 2024 validation",
        "constraint_satisfied": bool(feasible),
        "selected_variant": selected,
        "selected_validation_metrics": variants[selected],
        "adamic_adar_validation_metrics": graph_baseline,
        "validation_variants": variants,
        "limitations": [
            "Variant selection and reported scores use the same 2024 validation labels.",
            "The 2025 test was inspected previously and is not loaded by this training script.",
            "Hard negatives are unlabeled future pairs, not verified non-collaborations.",
            "Early-career and fairness groups use work-count proxies, not verified career stages.",
            "Do not deploy or claim superiority without a new untouched prospective holdout.",
        ],
    }
    artifact = {
        "model_name": report["model_name"],
        "feature_names": FEATURES,
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fairness_bonus": fairness_bonus,
        "semantic_slots": semantic_slots,
        "train_snapshot_sha256": report["train_snapshot_sha256"],
        "fit_features_through": 2022,
        "fit_labels_year": 2023,
        "validation_year": 2024,
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train abstract-aware graph-text ranker without reading 2025 test labels.")
    parser.add_argument("--train", type=Path, default=Path("data/multifield/frozen/train_asof_2023.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/multifield/frozen/validation_new_coauthors.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("data/models/temporal_graph_text_v2.json"))
    parser.add_argument("--report", type=Path, default=Path("data/evaluation/temporal_graph_text_v2.json"))
    parser.add_argument("--max-authors", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_v2(args.train, args.validation, args.model, args.report,
                      args.max_authors, args.max_queries, args.k, args.seed)
    print(json.dumps({"model": str(args.model), "report": str(args.report),
                      "selected_variant": result["selected_variant"],
                      "constraint_satisfied": result["constraint_satisfied"],
                      "selected_validation_metrics": result["selected_validation_metrics"],
                      "adamic_adar_validation_metrics": result["adamic_adar_validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
