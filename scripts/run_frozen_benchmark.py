from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from scripts.evaluate_collaborators import evaluate


def run(root: Path, output: Path, k: int = 10, max_authors: int = 3000, max_queries: int = 100,
        seed: int = 42, graphsage_embeddings: Path | None = None) -> dict:
    train = root / "train_asof_2023.jsonl"
    validation = root / "validation_new_coauthors.jsonl"
    test = root / "test_new_coauthors.jsonl"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["train_through"] != 2023 or manifest["validation_year"] != 2024 or manifest["test_year"] != 2025:
        raise ValueError("This benchmark requires the frozen 2023/2024/2025 protocol")

    settings = dict(k=k, max_authors=max_authors, max_queries=max_queries, seed=seed,
                    graphsage_embeddings_path=graphsage_embeddings)
    validation_result = evaluate(train, validation, **settings)
    selected = max(validation_result["models"],
                   key=lambda name: (validation_result["models"][name]["ndcg_at_k"], name))
    test_result = evaluate(train, test, prior_labels_path=validation, **settings)
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "fit on <=2023 publication-year proxy; select model by 2024 NDCG@K; report untouched 2025 test",
        "frozen_manifest": str(root / "manifest.json"),
        "frozen_artifact_hashes": manifest["artifacts"],
        "selected_on_validation": selected,
        "selection_metric": "ndcg_at_k",
        "validation": validation_result,
        "test": test_result,
        "selected_test_metrics": test_result["models"][selected],
        "limitations": [
            "Query authors have at least one held-out link in the sampled candidate pool; results do not generalize to all researchers.",
            "OpenAlex records were retrieved after publication, so this is not an archival historical snapshot.",
            "The graph model is compared only when explicitly supplied as train-only embeddings.",
            "Offline coauthorship ranking does not measure causal profile-promotion impact.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select on frozen validation labels and report the final test once.")
    parser.add_argument("--root", type=Path, default=Path("data/multifield/frozen"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/frozen_benchmark.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-authors", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graphsage-embeddings", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.output, args.k, args.max_authors, args.max_queries,
                 args.seed, args.graphsage_embeddings)
    print(json.dumps({"selected_on_validation": result["selected_on_validation"],
                      "validation_metrics": result["validation"]["models"][result["selected_on_validation"]],
                      "selected_test_metrics": result["selected_test_metrics"],
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
