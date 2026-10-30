from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from scripts.evaluate_collaborators import (
    _profiles_and_edges,
    _read_works,
    _text,
    _two_hop_scores,
    evaluate,
    ranking_metrics,
)


FEATURES = (
    "tfidf_cosine",
    "log_common_neighbors",
    "log_adamic_adar",
    "log_candidate_train_works",
    "log_candidate_train_degree",
    "low_work_count_proxy",
)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _future_edges(works: list[dict]) -> dict[str, set[str]]:
    _, edges = _profiles_and_edges(works)
    return edges


def _pair_labels(path: Path, expected_year: int = 2024) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for record in _read_works(path):
        if int(record.get("publication_year") or 0) != expected_year:
            raise ValueError(f"Expected only {expected_year} frozen labels")
        left, right = record["author_a"], record["author_b"]
        edges[left].add(right)
        edges[right].add(left)
    return dict(edges)


def _abstract(index: dict | None) -> str:
    if not index:
        return ""
    positioned = [(position, token) for token, positions in index.items() for position in positions]
    return " ".join(token for _, token in sorted(positioned))


class Period:
    def __init__(self, works: list[dict], max_authors: int, seed: int, use_abstracts: bool = False) -> None:
        self.profiles, self.neighbors = _profiles_and_edges(works)
        eligible = sorted(author for author, profile in self.profiles.items() if profile["titles"])
        if len(eligible) > max_authors:
            rng = random.Random(seed)
            popular = sorted(eligible, key=lambda author: (-self.profiles[author]["works"], author))[:max_authors // 2]
            sampled = rng.sample(sorted(set(eligible) - set(popular)), max_authors - len(popular))
            eligible = sorted([*popular, *sampled])
        self.authors = eligible
        self.index = {author: position for position, author in enumerate(eligible)}
        if use_abstracts:
            snippets: dict[str, list[str]] = defaultdict(list)
            for work in sorted(works, key=lambda item: (item["publication_year"], item.get("publication_date") or ""), reverse=True):
                abstract = _abstract(work.get("abstract_inverted_index"))[:1200]
                snippet = " ".join(part for part in (work.get("title") or "", abstract) if part).strip()
                if not snippet:
                    continue
                for item in work.get("authorships") or []:
                    author = ((item.get("author") or {}).get("id") or "").split("/")[-1]
                    if author in self.index and len(snippets[author]) < 4:
                        snippets[author].append(snippet)
            texts = [" ".join(snippets[author]) or _text(self.profiles[author]) for author in eligible]
        else:
            texts = [_text(self.profiles[author]) for author in eligible]
        self.tfidf = normalize(TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts))
        self.works = np.asarray([self.profiles[author]["works"] for author in eligible], dtype=np.float64)
        self.degrees = np.asarray([len(self.neighbors.get(author, set())) for author in eligible], dtype=np.float64)
        self.low_threshold = sorted(self.works)[(len(self.works) - 1) // 2]
        self._feature_cache: dict[str, np.ndarray] = {}

    def features(self, author: str) -> np.ndarray:
        if author in self._feature_cache:
            return self._feature_cache[author]
        lexical = np.asarray((self.tfidf @ self.tfidf[self.index[author]].T).toarray()).ravel()
        common, adamic = _two_hop_scores(author, self.authors, self.neighbors)
        result = np.column_stack((
            lexical,
            np.log1p(common),
            np.log1p(adamic),
            np.log1p(self.works),
            np.log1p(self.degrees),
            (self.works <= self.low_threshold).astype(np.float64),
        ))
        self._feature_cache[author] = result
        return result

    def labels(self, future: dict[str, set[str]]) -> dict[str, set[str]]:
        candidate_set = set(self.authors)
        return {
            author: (links - self.neighbors.get(author, set())) & (candidate_set - {author})
            for author, links in future.items() if author in candidate_set
        }

    def queries(self, labels: dict[str, set[str]], limit: int, seed: int) -> list[str]:
        queries = sorted(author for author, positives in labels.items() if positives)
        random.Random(seed).shuffle(queries)
        return queries[:limit]


def _training_pairs(period: Period, labels: dict[str, set[str]], queries: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for author in queries:
        features = period.features(author)
        positives = {period.index[item] for item in labels[author]}
        excluded = {period.index[item] for item in period.neighbors.get(author, set()) if item in period.index}
        excluded.add(period.index[author])
        candidates = set(range(len(period.authors))) - excluded - positives
        hard = set(np.argsort(features[:, 0])[-30:]) | set(np.argsort(features[:, 2])[-30:])
        negative = (hard & candidates) | set(rng.sample(sorted(candidates), min(50, len(candidates))))
        for position in sorted(positives):
            rows.append(features[position])
            targets.append(1)
        for position in sorted(negative):
            rows.append(features[position])
            targets.append(0)
    if not rows or len(set(targets)) != 2:
        raise ValueError("Training needs positive and negative query-candidate examples")
    return np.stack(rows), np.asarray(targets)


def _rank(period: Period, labels: dict[str, set[str]], queries: list[str], model: LogisticRegression | dict | None,
          scaler: StandardScaler | None, k: int, fairness_bonus: float, semantic_rescue: float = 0.0,
          semantic_slots: int = 0) -> dict[str, float]:
    totals = defaultdict(float)
    recommended: set[str] = set()
    exposure = defaultdict(float)
    low_group = {author for author in period.authors if period.profiles[author]["works"] <= period.low_threshold}
    for author in queries:
        features = period.features(author)
        if isinstance(model, dict):
            standardized = (features - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
            scores = standardized @ np.asarray(model["coefficient"]) + float(model["intercept"])
        elif model is not None and scaler is not None:
            scores = model.decision_function(scaler.transform(features))
        else:
            scores = features[:, 2].copy()
        scores += fairness_bonus * features[:, 5]
        scores += semantic_rescue * features[:, 0] * (features[:, 1] == 0)
        excluded = period.neighbors.get(author, set()) | {author}
        candidates = [item for item in period.authors if item not in excluded]
        ordered = sorted(candidates, key=lambda item: (-scores[period.index[item]], item))
        ranked = ordered[:k - semantic_slots]
        if semantic_slots:
            seen = set(ranked)
            topical = sorted((item for item in candidates if item not in seen and features[period.index[item], 0] > 0),
                             key=lambda item: (-features[period.index[item], 0], item))
            ranked.extend(topical[:semantic_slots])
            seen = set(ranked)
            ranked.extend(item for item in ordered if item not in seen and len(ranked) < k)
        for metric, value in ranking_metrics(ranked, labels[author], k).items():
            totals[metric] += value
        recommended.update(ranked)
        for rank, item in enumerate(ranked):
            exposure[item] += 1 / math.log2(rank + 2)
    denominator = len(queries)
    low_exposure = sum(exposure[item] for item in low_group) / (len(low_group) * denominator)
    high_group = set(period.authors) - low_group
    high_exposure = sum(exposure[item] for item in high_group) / (max(len(high_group), 1) * denominator)
    return {
        **{name: round(value / denominator, 6) for name, value in totals.items()},
        "coverage": round(len(recommended) / len(period.authors), 6),
        "fairness_exposure_gap": round(abs(low_exposure - high_exposure), 6),
    }


def train(train_path: Path, validation_path: Path, model_path: Path, report_path: Path,
          max_authors: int = 3000, max_queries: int = 100, k: int = 10, seed: int = 42) -> dict:
    works = _read_works(train_path)
    years = {int(work["publication_year"]) for work in works}
    if not years or min(years) < 2020 or max(years) != 2023:
        raise ValueError("Expected a frozen 2020-2023 train snapshot")
    past = [work for work in works if work["publication_year"] <= 2022]
    year_2023 = [work for work in works if work["publication_year"] == 2023]
    training_period = Period(past, max_authors, seed)
    training_labels = training_period.labels(_future_edges(year_2023))
    training_queries = training_period.queries(training_labels, max_queries, seed)
    x, y = _training_pairs(training_period, training_labels, training_queries, seed)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed).fit(scaler.transform(x), y)

    validation_period = Period(works, max_authors, seed)
    validation_labels = validation_period.labels(_pair_labels(validation_path))
    validation_queries = validation_period.queries(validation_labels, max_queries, seed)
    if not validation_queries:
        raise ValueError("No validation links in the train-only candidate pool")
    variants = {str(bonus): _rank(validation_period, validation_labels, validation_queries, model, scaler, k, bonus)
                for bonus in (0.0, 0.05, 0.1)}
    graph_baseline = _rank(validation_period, validation_labels, validation_queries, None, None, k, 0.0)
    selected_bonus = max(variants, key=lambda bonus: (
        variants[bonus]["ndcg_at_k"], -variants[bonus]["fairness_exposure_gap"], -float(bonus)))
    baselines = evaluate(train_path, validation_path, k=k, max_authors=max_authors,
                         max_queries=max_queries, seed=seed)["models"]
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "TemporalGraphTextRanker-v1",
        "train_snapshot": str(train_path),
        "train_snapshot_sha256": _digest(train_path),
        "validation_labels": str(validation_path),
        "validation_labels_sha256": _digest(validation_path),
        "feature_period_for_fit": "2020-2022",
        "supervision_period_for_fit": 2023,
        "feature_period_for_validation": "2020-2023",
        "validation_year": 2024,
        "test_year": None,
        "candidate_authors_fit": len(training_period.authors),
        "candidate_authors_validation": len(validation_period.authors),
        "positive_link_queries_fit": len(training_queries),
        "positive_link_queries_validation": len(validation_queries),
        "positive_training_examples": int(y.sum()),
        "negative_training_examples": int(len(y) - y.sum()),
        "k": k, "seed": seed,
        "fairness_bonus_selected_on_validation": float(selected_bonus),
        "validation_graph_text_variants": variants,
        "validation_baselines": {**{name: baselines[name] for name in ("tfidf", "hybrid")},
                                 "graph_adamic_adar": graph_baseline},
        "limitations": [
            "2024 validation selected the fairness bonus; its score is model-selection evidence, not an untouched final estimate.",
            "The 2025 test was inspected in earlier experiments and is not used here.",
            "Queries are conditional on an observed future link in the sampled candidate pool.",
            "Missing future coauthorships are unlabeled rather than confirmed negatives.",
            "A new prospective holdout is required before claiming superiority or deploying this model.",
        ],
    }
    artifact = {
        "model_name": report["model_name"],
        "feature_names": FEATURES,
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fairness_bonus": float(selected_bonus),
        "train_snapshot_sha256": report["train_snapshot_sha256"],
        "training_cutoff_year": 2022,
        "label_year": 2023,
        "validation_year": 2024,
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit graph-text weights on 2023 links; validate only on 2024.")
    parser.add_argument("--train", type=Path, default=Path("data/multifield/frozen/train_asof_2023.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/multifield/frozen/validation_new_coauthors.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("data/models/temporal_graph_text_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("data/evaluation/temporal_graph_text_v1.json"))
    parser.add_argument("--max-authors", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(args.train, args.validation, args.model, args.report,
                   args.max_authors, args.max_queries, args.k, args.seed)
    print(json.dumps({"model": str(args.model), "report": str(args.report),
                      "fit_examples": result["positive_training_examples"] + result["negative_training_examples"],
                      "selected_fairness_bonus": result["fairness_bonus_selected_on_validation"],
                      "validation_graph_text": result["validation_graph_text_variants"],
                      "validation_baselines": result["validation_baselines"]}, indent=2))


if __name__ == "__main__":
    main()
