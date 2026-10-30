from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean, median

import numpy as np
from sklearn.preprocessing import normalize

from scripts.evaluate_collaborators import _graph_neighbors, _read_works


def diagnose(graph_dir: Path, validation_labels: Path, embeddings_path: Path, seed: int = 42) -> dict:
    nodes = json.loads((graph_dir / "author_nodes.json").read_text(encoding="utf-8"))
    neighbors = _graph_neighbors(graph_dir / "author_edges.csv")
    author_ids = {node["author_id"] for node in nodes}
    degrees = [len(neighbors.get(author_id, set())) for author_id in author_ids]
    labels = _read_works(validation_labels)
    if not labels or any(int(item.get("publication_year") or 0) != 2024 for item in labels):
        raise ValueError("Graph diagnostics require only frozen 2024 validation labels")
    with np.load(embeddings_path) as saved:
        embedding_ids = [str(value) for value in saved["author_ids"]]
        if set(embedding_ids) != author_ids:
            raise ValueError("Embedding author IDs do not match the train-only graph")
        vectors = normalize(saved["embeddings"])
    index = {author_id: position for position, author_id in enumerate(embedding_ids)}
    global_rng = np.random.default_rng(seed)
    random_left = global_rng.integers(0, len(vectors), size=len(labels))
    random_right = global_rng.integers(0, len(vectors), size=len(labels))
    global_random_scores = np.sum(vectors[random_left] * vectors[random_right], axis=1)
    positives = [(item["author_a"], item["author_b"]) for item in labels
                 if item["author_a"] in author_ids and item["author_b"] in author_ids]
    if not positives:
        raise ValueError("No validation pairs belong to the training graph")

    rng = random.Random(seed)
    population = sorted(author_ids)
    positive_scores = []
    random_scores = []
    common_neighbor_counts = []
    endpoint_degrees = []
    missed_examples = []
    for left, right in positives:
        score = float(vectors[index[left]] @ vectors[index[right]])
        common = len(neighbors.get(left, set()) & neighbors.get(right, set()))
        positive_scores.append(score)
        common_neighbor_counts.append(common)
        endpoint_degrees.extend((len(neighbors.get(left, set())), len(neighbors.get(right, set()))))
        if common == 0 and len(missed_examples) < 5:
            missed_examples.append({"author_a": left, "author_b": right,
                                    "train_degrees": [len(neighbors.get(left, set())), len(neighbors.get(right, set()))],
                                    "graphsage_similarity": round(score, 6)})
        for _ in range(100):
            candidate = population[rng.randrange(len(population))]
            if candidate != left and candidate not in neighbors.get(left, set()) and candidate != right:
                random_scores.append(float(vectors[index[left]] @ vectors[index[candidate]]))
                break
    if len(random_scores) != len(positive_scores):
        raise ValueError("Could not draw one unlabeled comparison pair per positive")

    counts = [int(node.get("works_count", 0) or 0) for node in nodes]
    return {
        "training_graph": str(graph_dir),
        "validation_labels": str(validation_labels),
        "embeddings": str(embeddings_path),
        "seed": seed,
        "train_authors": len(author_ids),
        "train_directed_edges": sum(degrees),
        "authors_with_no_train_coauthors": sum(degree == 0 for degree in degrees),
        "median_train_degree": median(degrees),
        "median_train_works_per_author": median(counts),
        "validation_pairs": len(positives),
        "validation_pairs_with_common_train_neighbor": sum(count > 0 for count in common_neighbor_counts),
        "validation_pairs_with_two_isolated_endpoints": sum(
            not neighbors.get(left) and not neighbors.get(right) for left, right in positives),
        "median_validation_endpoint_train_degree": median(endpoint_degrees),
        "mean_validation_pair_cosine": round(mean(positive_scores), 6),
        "mean_random_unlabeled_pair_cosine": round(mean(random_scores), 6),
        "mean_global_random_pair_cosine": round(float(global_random_scores.mean()), 6),
        "global_random_pairs_with_cosine_at_least_0_999": int(np.sum(global_random_scores >= 0.999)),
        "validation_pair_beats_matched_random_fraction": round(mean(
            positive > random_score for positive, random_score in zip(positive_scores, random_scores, strict=True)), 6),
        "examples_without_common_neighbor": missed_examples,
        "caveats": [
            "Only 2024 validation labels were inspected; 2025 test labels were not used for diagnosis.",
            "Randomly sampled nonedges are unlabeled, not verified non-collaborations.",
            "Global random-pair AUC and query-matched comparisons have different base rates and are not interchangeable.",
            "Train-period author IDs and graph features are a publication-year proxy, not an archival historical snapshot.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose train-only GraphSAGE features and 2024 validation failures.")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/multifield/frozen/graphsage_train"))
    parser.add_argument("--validation-labels", type=Path, default=Path("data/multifield/frozen/validation_new_coauthors.jsonl"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/multifield/frozen/graphsage_model/graphsage_author_embeddings.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/graphsage_diagnostics.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = diagnose(args.graph_dir, args.validation_labels, args.embeddings, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
