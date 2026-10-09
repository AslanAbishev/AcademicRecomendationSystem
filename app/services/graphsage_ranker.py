from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _short_id(value: str | None) -> str:
    return value.rstrip("/").split("/")[-1] if value else ""


class GraphSAGERanker:
    """Load trained GraphSAGE artifacts and score transductive or new live authors."""

    def __init__(self, embeddings_path: Path, model_path: Path) -> None:
        self.embeddings_path = embeddings_path
        self.model_path = model_path
        self.embeddings: dict[str, np.ndarray] | None = None
        self.model = None
        self.feature_config: dict[str, Any] | None = None
        self.status = "GraphSAGE artifacts are not loaded."

    def score_pair(
        self,
        author: dict | None,
        candidate: dict,
        context_works: list[dict] | None = None,
    ) -> tuple[float | None, str]:
        author_id = _short_id((author or {}).get("id"))
        candidate_id = _short_id(candidate.get("id"))
        if not author_id or not candidate_id:
            return None, "unavailable"

        embeddings = self._load_embeddings()
        if author_id in embeddings and candidate_id in embeddings:
            score = float(np.dot(embeddings[author_id], embeddings[candidate_id]))
            return round(float(np.clip((score + 1.0) / 2.0, 0.0, 1.0)), 4), "transductive"

        score = self._score_inductive(author, candidate, context_works or [])
        if score is None:
            return None, "outside_training_graph"
        return round(score, 4), "inductive"

    def prepare(self) -> None:
        self._load_embeddings()
        self._load_model()

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if self.embeddings is not None:
            return self.embeddings
        self.embeddings = {}
        try:
            with np.load(self.embeddings_path, allow_pickle=False) as payload:
                author_ids = payload["author_ids"]
                vectors = np.asarray(payload["embeddings"], dtype="float32")
            if len(author_ids) != len(vectors) or vectors.ndim != 2:
                raise ValueError("author_ids and embeddings have incompatible shapes")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = vectors / np.maximum(norms, 1e-8)
            self.embeddings = {
                _short_id(str(author_id)): normalized[index]
                for index, author_id in enumerate(author_ids)
                if _short_id(str(author_id))
            }
            self.status = f"GraphSAGE live ranking active: {len(self.embeddings)} author embeddings loaded."
        except Exception as exc:
            self.status = f"GraphSAGE artifacts unavailable: {exc}"
        return self.embeddings

    def _load_model(self):
        if self.model is not None:
            return self.model
        try:
            import torch
            from torch_geometric.nn import SAGEConv

            checkpoint = torch.load(self.model_path, map_location="cpu")
            hidden_dim = int(checkpoint["hidden_dim"])
            embedding_dim = int(checkpoint["embedding_dim"])
            feature_dim = int(checkpoint["feature_dim"])

            class GraphSAGE(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv1 = SAGEConv(feature_dim, hidden_dim)
                    self.conv2 = SAGEConv(hidden_dim, embedding_dim)

                def forward(self, features, edges):
                    return self.conv2(self.conv1(features, edges).relu(), edges)

            model = GraphSAGE()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.model = model
            self.feature_config = checkpoint.get("feature_config") or self._fallback_feature_config()
            if self.embeddings:
                self.status = (
                    f"GraphSAGE live ranking active: {len(self.embeddings)} stored embeddings "
                    "plus inductive inference for unseen authors."
                )
            return self.model
        except Exception as exc:
            if self.embeddings:
                self.status = (
                    f"GraphSAGE transductive ranking active for {len(self.embeddings)} authors; "
                    f"inductive inference unavailable ({exc})."
                )
            else:
                self.status = f"GraphSAGE artifacts unavailable: {exc}"
            return None

    def _score_inductive(self, author: dict, candidate: dict, context_works: list[dict]) -> float | None:
        model = self._load_model()
        if model is None:
            return None
        try:
            import torch

            nodes: dict[str, dict] = {}
            for item in (author, candidate):
                self._add_node(nodes, item)
            edges: set[tuple[str, str]] = set()
            for work in context_works:
                work_authors = []
                for authorship in work.get("authorships", []):
                    payload = authorship.get("author") or {}
                    author_id = _short_id(payload.get("id"))
                    if not author_id:
                        continue
                    node = {
                        **payload,
                        "country_code": (authorship.get("countries") or [None])[0],
                        "last_known_institutions": authorship.get("institutions") or [],
                    }
                    self._add_node(nodes, node)
                    work_authors.append(author_id)
                for left, right in combinations(sorted(set(work_authors)), 2):
                    edges.add((left, right))
                    edges.add((right, left))

            node_ids = list(nodes)
            node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
            features = np.array([self._node_features(nodes[node_id]) for node_id in node_ids], dtype="float32")
            edge_array = np.array(
                [[node_to_index[left], node_to_index[right]] for left, right in edges],
                dtype="int64",
            ).T if edges else np.empty((2, 0), dtype="int64")
            with torch.no_grad():
                vectors = model(
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(edge_array, dtype=torch.long),
                )
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy()
            left = vectors[node_to_index[_short_id(author["id"])]]
            right = vectors[node_to_index[_short_id(candidate["id"])]]
            cosine = float(np.dot(left, right))
            return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        except Exception:
            return None

    def _add_node(self, nodes: dict[str, dict], payload: dict) -> None:
        author_id = _short_id(payload.get("id"))
        if author_id:
            nodes.setdefault(author_id, payload)

    def _node_features(self, item: dict) -> list[float]:
        config = self.feature_config or self._fallback_feature_config()
        institutions = item.get("last_known_institutions") or item.get("institutions") or []
        country = item.get("country_code") or (institutions[0].get("country_code") if institutions else "") or ""
        country_index = config.get("country_index", {})
        country_count = max(int(config.get("country_count", 1)), 1)
        works = int(item.get("works_count", 0) or 0) / max(int(config.get("max_works", 1)), 1)
        citations = int(item.get("cited_by_count", 0) or 0) / max(int(config.get("max_citations", 1)), 1)
        country_value = country_index.get(country, 0) / country_count
        ecr_signal = 1.0 if int(item.get("works_count", 0) or 0) <= 25 else 0.0
        return [works, citations, country_value, ecr_signal]

    def _fallback_feature_config(self) -> dict[str, Any]:
        return {"max_works": 1, "max_citations": 1, "country_index": {}, "country_count": 1}
