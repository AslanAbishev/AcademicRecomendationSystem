from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


def _read_works(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _author_ids(work: dict) -> list[str]:
    return sorted({
        (authorship.get("author") or {}).get("id", "").split("/")[-1]
        for authorship in work.get("authorships") or []
        if (authorship.get("author") or {}).get("id")
    })


def _profiles_and_edges(works: list[dict]) -> tuple[dict[str, dict], dict[str, set[str]]]:
    profiles: dict[str, dict] = defaultdict(lambda: {"titles": [], "topics": [], "works": 0})
    edges: dict[str, set[str]] = defaultdict(set)
    for work in works:
        ids = _author_ids(work)
        title = work.get("title") or ""
        topics = [item.get("display_name") or "" for item in work.get("topics") or []][:3]
        for author_id in ids:
            profiles[author_id]["works"] += 1
            if title:
                profiles[author_id]["titles"].append(title)
            profiles[author_id]["topics"].extend(topics)
            edges[author_id].update(other for other in ids if other != author_id)
    return dict(profiles), dict(edges)


def _label_edges(records: list[dict]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for record in records:
        left, right = record["author_a"], record["author_b"]
        edges[left].add(right)
        edges[right].add(left)
    return dict(edges)


def _graph_neighbors(path: Path) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            left, right = row["source"], row["target"]
            if left != right:
                neighbors[left].add(right)
                neighbors[right].add(left)
    return dict(neighbors)


def _two_hop_scores(author_id: str, candidates: list[str], neighbors: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray]:
    common = Counter()
    adamic = Counter()
    for intermediary in neighbors.get(author_id, set()):
        weight = 1 / math.log1p(max(len(neighbors.get(intermediary, set())), 2))
        for candidate in neighbors.get(intermediary, set()):
            if candidate != author_id:
                common[candidate] += 1
                adamic[candidate] += weight
    return (np.asarray([common[item] for item in candidates], dtype=np.float64),
            np.asarray([adamic[item] for item in candidates], dtype=np.float64))


def _text(profile: dict) -> str:
    return " ".join([*profile["titles"][:8], *profile["topics"][:16]]).strip() or "unknown research topic"


def ranking_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    hits = [1 if item in relevant else 0 for item in top]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    return {
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(hits) / len(relevant) if relevant else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def _bm25(query: str, postings: dict[str, list[tuple[int, int]]], df: Counter, lengths: np.ndarray) -> np.ndarray:
    score = np.zeros(len(lengths), dtype=np.float64)
    average_length = float(lengths.mean()) or 1.0
    for term in set(re.findall(r"[a-z0-9]{3,}", query.lower())):
        frequency = df.get(term, 0)
        if not frequency:
            continue
        idf = math.log1p((len(lengths) - frequency + 0.5) / (frequency + 0.5))
        for index, count in postings.get(term, []):
            score[index] += idf * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * lengths[index] / average_length))
    maximum = score.max(initial=0)
    return score / maximum if maximum else score


def evaluate(train_path: Path, test_path: Path, k: int = 10, max_authors: int = 3000, max_queries: int = 100, seed: int = 42, prior_labels_path: Path | None = None, graphsage_embeddings_path: Path | None = None, graph_edges_path: Path | None = None) -> dict:
    if k < 1 or max_authors < k + 1 or max_queries < 1:
        raise ValueError("k and max_queries must be positive; max_authors must exceed k")
    train = _read_works(train_path)
    test = _read_works(test_path)
    train_years = {int(work["publication_year"]) for work in train if work.get("publication_year")}
    test_years = {int(work["publication_year"]) for work in test if work.get("publication_year")}
    if not train_years or not test_years or max(train_years) >= min(test_years):
        raise ValueError("Evaluation needs strictly later test years than training years")
    profiles, known_edges = _profiles_and_edges(train)
    label_records = bool(test and "author_a" in test[0] and "author_b" in test[0])
    _, future_edges = (None, _label_edges(test)) if label_records else _profiles_and_edges(test)
    prior_edges = _label_edges(_read_works(prior_labels_path)) if prior_labels_path else {}
    all_future_links = sum(
        len((neighbors - known_edges.get(author_id, set()) - prior_edges.get(author_id, set())) & (set(profiles) - {author_id}))
        for author_id, neighbors in future_edges.items() if author_id in profiles
    )
    eligible = sorted(author_id for author_id, profile in profiles.items() if profile["titles"])
    if len(eligible) > max_authors:
        rng = random.Random(seed)
        popular = sorted(eligible, key=lambda author_id: (-profiles[author_id]["works"], author_id))[: max_authors // 2]
        sampled = rng.sample(sorted(set(eligible) - set(popular)), max_authors - len(popular))
        eligible = sorted([*popular, *sampled])
    eligible_set = set(eligible)
    labels = {
        author_id: (neighbors - known_edges.get(author_id, set()) - prior_edges.get(author_id, set())) & (eligible_set - {author_id})
        for author_id, neighbors in future_edges.items()
        if author_id in eligible_set
    }
    queries = sorted(author_id for author_id, positives in labels.items() if positives)
    if not queries:
        raise ValueError("No held-out new coauthor links connect eligible training authors")
    random.Random(seed).shuffle(queries)
    queries = queries[:max_queries]
    evaluated_links = sum(len(labels[author_id]) for author_id in queries)

    texts = [_text(profiles[author_id]) for author_id in eligible]
    tfidf = normalize(TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts))
    hashing = HashingVectorizer(n_features=4096, alternate_sign=False, stop_words="english", ngram_range=(1, 2)).transform(texts)
    components = min(64, *tfidf.shape) - 1
    latent = normalize(TruncatedSVD(n_components=components, random_state=seed).fit_transform(tfidf)) if components > 0 else None
    tokens = [re.findall(r"[a-z0-9]{3,}", text.lower()) for text in texts]
    df = Counter(term for document in tokens for term in set(document))
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, document in enumerate(tokens):
        for term, frequency in Counter(document).items():
            postings[term].append((position, frequency))
    lengths = np.asarray([max(len(document), 1) for document in tokens])
    index = {author_id: position for position, author_id in enumerate(eligible)}
    graph_vectors = None
    if graphsage_embeddings_path:
        with np.load(graphsage_embeddings_path) as saved:
            graph_index = {str(author_id): position for position, author_id in enumerate(saved["author_ids"])}
            if not eligible_set <= graph_index.keys():
                raise ValueError("GraphSAGE checkpoint does not cover the frozen candidate pool")
            graph_vectors = normalize(saved["embeddings"][[graph_index[author_id] for author_id in eligible]])
    graph_neighbors = _graph_neighbors(graph_edges_path) if graph_edges_path else None
    work_counts = sorted(profiles[author_id]["works"] for author_id in eligible)
    low_threshold = work_counts[(len(work_counts) - 1) // 2]
    low_group = {author_id for author_id in eligible if profiles[author_id]["works"] <= low_threshold}
    accumulators = defaultdict(lambda: {"metrics": Counter(), "recommended": set(), "exposure": Counter()})

    for author_id in queries:
        query_index = index[author_id]
        lexical = np.asarray((tfidf @ tfidf[query_index].T).toarray()).ravel()
        semantic = np.asarray((hashing @ hashing[query_index].T).toarray()).ravel()
        lsa = latent @ latent[query_index] if latent is not None else lexical
        bm25 = _bm25(texts[query_index], postings, df, lengths)
        random_generator = random.Random(f"{seed}:{author_id}")
        hybrid = 0.23 * lexical + 0.2 * bm25 + 0.25 * lsa + 0.32 * semantic
        low_visibility_bonus = np.asarray([1.0 if item in low_group else 0.0 for item in eligible])
        scores = {
            "random": np.asarray([random_generator.random() for _ in eligible]),
            "popularity": np.asarray([profiles[item]["works"] for item in eligible], dtype=np.float64),
            "tfidf": lexical,
            "bm25": bm25,
            "lsa_svd": lsa,
            "semantic_hashing": semantic,
            "hybrid": hybrid,
            "fairness_reranked_hybrid": 0.9 * hybrid + 0.1 * low_visibility_bonus,
        }
        if graph_vectors is not None:
            scores["graphsage"] = graph_vectors @ graph_vectors[query_index]
        if graph_neighbors is not None:
            common, adamic = _two_hop_scores(author_id, eligible, graph_neighbors)
            scores["graph_common_neighbors"] = common
            scores["graph_adamic_adar"] = adamic
        excluded = known_edges.get(author_id, set()) | prior_edges.get(author_id, set()) | {author_id}
        candidates = [item for item in eligible if item not in excluded]
        for name, model_scores in scores.items():
            ranked = sorted(candidates, key=lambda item: (-model_scores[index[item]], item))[:k]
            measured = ranking_metrics(ranked, labels[author_id], k)
            bucket = accumulators[name]
            bucket["metrics"].update(measured)
            bucket["recommended"].update(ranked)
            for rank, item in enumerate(ranked):
                bucket["exposure"][item] += 1 / math.log2(rank + 2)

    denominator = len(queries)
    results = {}
    for name, bucket in accumulators.items():
        low_exposure = sum(bucket["exposure"][item] for item in low_group) / (len(low_group) * denominator)
        high_exposure = sum(bucket["exposure"][item] for item in eligible_set - low_group) / (max(len(eligible_set - low_group), 1) * denominator)
        results[name] = {
            **{metric: round(value / denominator, 6) for metric, value in bucket["metrics"].items()},
            "coverage": round(len(bucket["recommended"]) / len(eligible), 6),
            "fairness_exposure_gap": round(abs(low_exposure - high_exposure), 6),
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "train": str(train_path), "test": str(test_path),
        "train_works": len(train), "test_works": len(test),
        "train_years": sorted(train_years), "test_years": sorted(test_years),
        "k": k, "seed": seed, "candidate_authors": len(eligible), "evaluated_authors": denominator,
        "known_train_authors": len(profiles), "future_new_directed_links_in_full_train_pool": all_future_links,
        "held_out_links_in_evaluated_queries": evaluated_links,
        "ground_truth": "new test-period coauthors who were present in training; previous coauthors excluded",
        "label_records": label_records,
        "prior_labels": str(prior_labels_path) if prior_labels_path else None,
        "graphsage_embeddings": str(graphsage_embeddings_path) if graphsage_embeddings_path else None,
        "train_graph_edges": str(graph_edges_path) if graph_edges_path else None,
        "fairness_groups": f"low visibility proxy: at most {low_threshold} train-period works; not a verified career-stage label",
        "fairness_exposure_gap_definition": "absolute difference in mean discounted top-K exposure per eligible low/high group author per query",
        "coverage_definition": "unique recommended authors divided by eligible training authors",
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare leakage-safe collaborator rankers on future OpenAlex coauthorships.")
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--prior-labels", help="Earlier-period pairs to exclude from later evaluation candidates")
    parser.add_argument("--graphsage-embeddings", help="Train-only GraphSAGE author embeddings for the same candidate pool")
    parser.add_argument("--train-graph-edges", help="Train-only author_edges.csv for graph baselines")
    parser.add_argument("--output", default="data/evaluation/collaborators.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-authors", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate(Path(args.train), Path(args.test), args.k, args.max_authors, args.max_queries, args.seed, Path(args.prior_labels) if args.prior_labels else None, Path(args.graphsage_embeddings) if args.graphsage_embeddings else None, Path(args.train_graph_edges) if args.train_graph_edges else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
