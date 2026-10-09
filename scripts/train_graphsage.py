from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def train_graphsage(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
        from torch_geometric.utils import negative_sampling
    except ImportError as exc:
        raise RuntimeError(
            "GraphSAGE training requires optional ML dependencies. "
            "Install them with: pip install -r requirements-ml.txt"
        ) from exc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    graph_dir = Path(args.graph_dir)
    validation_graph_dir = Path(args.validation_graph_dir) if args.validation_graph_dir else None
    test_graph_dir = Path(args.test_graph_dir) if args.test_graph_dir else None
    author_nodes = _merge_author_nodes(graph_dir, validation_graph_dir, test_graph_dir)
    author_to_index = {item["author_id"]: index for index, item in enumerate(author_nodes)}
    feature_config = _feature_config(author_nodes)
    features = _author_features(author_nodes, feature_config)
    edge_index = _edge_index(graph_dir / "author_edges.csv", author_to_index)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        def forward(self, x, edges):
            x = self.conv1(x, edges).relu()
            return self.conv2(x, edges)

    x = torch.tensor(features, dtype=torch.float32)
    edges = torch.tensor(edge_index, dtype=torch.long)
    model = GraphSAGE(x.shape[1], args.hidden_dim, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        z = model(x, edges)
        negative_edges = negative_sampling(edges, num_nodes=x.shape[0], num_neg_samples=edges.shape[1])
        positive_logits = (z[edges[0]] * z[edges[1]]).sum(dim=1)
        negative_logits = (z[negative_edges[0]] * z[negative_edges[1]]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        loss = loss + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        loss.backward()
        optimizer.step()
        if epoch % max(args.epochs // 10, 1) == 0:
            print(f"epoch={epoch} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = F.normalize(model(x, edges), p=2, dim=1).cpu().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train": _link_prediction_auc(
            embeddings,
            edge_index,
            edges,
            args.seed,
        ),
    }
    if validation_graph_dir:
        validation_edges = _edge_index(
            validation_graph_dir / "author_edges.csv",
            author_to_index,
            required=False,
        )
        metrics["validation"] = _link_prediction_auc(
            embeddings,
            validation_edges,
            edges,
            args.seed + 1,
        )
    if test_graph_dir:
        test_edges = _edge_index(
            test_graph_dir / "author_edges.csv",
            author_to_index,
            required=False,
        )
        metrics["test"] = _link_prediction_auc(
            embeddings,
            test_edges,
            edges,
            args.seed + 2,
        )
    np.savez_compressed(
        output_dir / "graphsage_author_embeddings.npz",
        author_ids=np.array([item["author_id"] for item in author_nodes]),
        embeddings=embeddings,
    )
    (output_dir / "graphsage_metadata.json").write_text(
        json.dumps(
            {
                "authors": len(author_nodes),
                "edges": int(edges.shape[1]),
                "epochs": args.epochs,
                "embedding_dim": args.embedding_dim,
                "objective": "co-author link prediction with negative sampling",
                "metrics": metrics,
                "training_graph": str(graph_dir),
                "validation_graph": str(validation_graph_dir) if validation_graph_dir else None,
                "test_graph": str(test_graph_dir) if test_graph_dir else None,
                "feature_config": feature_config,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "author_ids": [item["author_id"] for item in author_nodes],
            "feature_dim": int(x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "feature_config": feature_config,
        },
        output_dir / "graphsage_model.pt",
    )
    print(json.dumps({"status": "trained", "output_dir": str(output_dir), "metrics": metrics}, indent=2))


def _merge_author_nodes(*graph_dirs: Path | None) -> list[dict]:
    merged: dict[str, dict] = {}
    for graph_dir in graph_dirs:
        if graph_dir is None:
            continue
        path = graph_dir / "author_nodes.json"
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            merged.setdefault(item["author_id"], item)
    if not merged:
        raise FileNotFoundError("No author_nodes.json files were found for GraphSAGE.")
    return list(merged.values())


def _feature_config(author_nodes: list[dict]) -> dict:
    countries: dict[str, int] = {}
    for item in author_nodes:
        country = item.get("country_code") or ""
        if country not in countries:
            countries[country] = len(countries)
    return {
        "max_works": max([int(item.get("works_count", 0) or 0) for item in author_nodes] or [1], default=1),
        "max_citations": max([int(item.get("cited_by_count", 0) or 0) for item in author_nodes] or [1], default=1),
        "country_index": countries,
        "country_count": max(len(countries), 1),
    }


def _author_features(author_nodes: list[dict], feature_config: dict | None = None) -> np.ndarray:
    rows = []
    config = feature_config or _feature_config(author_nodes)
    max_works = int(config["max_works"])
    max_citations = int(config["max_citations"])
    countries = config["country_index"]
    country_count = int(config["country_count"])
    for item in author_nodes:
        works = int(item.get("works_count", 0) or 0) / max(max_works, 1)
        citations = int(item.get("cited_by_count", 0) or 0) / max(max_citations, 1)
        country = countries.get(item.get("country_code") or "", 0) / country_count
        ecr_signal = 1.0 if int(item.get("works_count", 0) or 0) <= 25 else 0.0
        rows.append([works, citations, country, ecr_signal])
    return np.array(rows, dtype="float32")


def _edge_index(
    path: Path,
    author_to_index: dict[str, int],
    required: bool = True,
) -> np.ndarray:
    edges: list[tuple[int, int]] = []
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Graph edge file was not found: {path}")
        return np.empty((2, 0), dtype="int64")
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = author_to_index.get(row["source"])
            target = author_to_index.get(row["target"])
            if source is None or target is None or source == target:
                continue
            edges.append((source, target))
    if not edges and required:
        raise ValueError("No co-author edges were found. Build a larger graph dataset first.")
    return np.array(edges, dtype="int64").T if edges else np.empty((2, 0), dtype="int64")


def _link_prediction_auc(
    embeddings: np.ndarray,
    positive_edges: np.ndarray,
    training_edges: np.ndarray,
    seed: int,
) -> float | None:
    if positive_edges.shape[1] == 0:
        return None
    positive_scores = _edge_scores(embeddings, positive_edges)
    rng = np.random.default_rng(seed)
    existing = {(int(left), int(right)) for left, right in training_edges.T}
    existing.update((int(left), int(right)) for left, right in positive_edges.T)
    negative_edges: list[tuple[int, int]] = []
    attempts = 0
    node_count = embeddings.shape[0]
    while len(negative_edges) < positive_edges.shape[1] and attempts < positive_edges.shape[1] * 20:
        left = int(rng.integers(0, node_count))
        right = int(rng.integers(0, node_count))
        attempts += 1
        if left == right or (left, right) in existing:
            continue
        negative_edges.append((left, right))
    if not negative_edges:
        return None
    negative_scores = _edge_scores(
        embeddings,
        np.array(negative_edges, dtype="int64").T,
    )
    pair_count = min(100_000, len(positive_scores) * len(negative_scores))
    positive_indices = rng.integers(0, len(positive_scores), size=pair_count)
    negative_indices = rng.integers(0, len(negative_scores), size=pair_count)
    sampled_positive = positive_scores[positive_indices]
    sampled_negative = negative_scores[negative_indices]
    comparisons = (sampled_positive > sampled_negative).mean()
    ties = (sampled_positive == sampled_negative).mean()
    return round(float(comparisons + 0.5 * ties), 4)


def _edge_scores(embeddings: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return (embeddings[edges[0]] * embeddings[edges[1]]).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GraphSAGE author embedding model on co-author graph data.")
    parser.add_argument("--graph-dir", default="data/graphsage")
    parser.add_argument("--validation-graph-dir")
    parser.add_argument("--test-graph-dir")
    parser.add_argument("--output-dir", default="data/graphsage")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_graphsage(args)


if __name__ == "__main__":
    main()
