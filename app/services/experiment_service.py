from __future__ import annotations

import json
from pathlib import Path

from app.schemas import ExperimentPlanResponse


class ExperimentService:
    def get_collaborator_evaluation(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def get_ml_plan(self) -> ExperimentPlanResponse:
        return ExperimentPlanResponse(
            dataset_source="OpenAlex",
            target_slice="Recommender systems, publication years 2020-2025; 17,677 collected unique works locally",
            models=[
                "Baseline: TF-IDF + cosine similarity",
                "Baseline: BM25 information retrieval ranker",
                "Baseline: LSA/SVD latent semantic model",
                "Local embedding fallback: semantic hashing vectors",
                "Main local model: TF-IDF + BM25 + LSA/SVD + semantic hashing ensemble",
                "Transformer embedding models: SciBERT and SPECTER2 stored in PostgreSQL/pgvector",
                "Graph layer: co-author/work/topic overlap score inspired by GraphSAGE features",
                "Fairness-aware reranking for early-career and low-citation candidates",
            ],
            features=[
                "title, abstract/indexed topics, venue, publication year",
                "author affiliation, country, works count, citation count, h-index",
                "OpenAlex topic IDs/names and recent works",
                "co-author evidence works and shared topic overlap",
                "profile completeness, connected academic identities, promotion actions",
            ],
            metrics=[
                "Precision@K and Recall@K for recommended works/collaborators",
                "NDCG@K for ranking quality",
                "MRR for first relevant collaborator/source",
                "Coverage and diversity across institutions/countries",
                "Fairness exposure gap between established and early-career researchers",
                "Promotion delta: change in visibility score, works, citations, h-index, and completed actions",
            ],
            validation_strategy=(
                "Temporal split: train on OpenAlex works up to year T, validate recommendations against "
                "future co-authorships, venues, and topic-consistent publications after T."
            ),
            notes=[
                "No synthetic researchers or fake opportunities are used in runtime recommendations.",
                "Runtime recommendations now expose per-model scores, so TF-IDF, BM25, LSA/SVD, semantic hashing, and the hybrid ensemble can be compared.",
                "SciBERT and SPECTER2 can be selected in the dashboard for pgvector-based paper recommendations.",
                "PostgreSQL + pgvector stores persistent embeddings and supports vector similarity retrieval.",
                "The local author-linked OpenAlex corpus contains 17,677 unique works; cohort seeds are selected from train years only.",
                "Crossref activity is stored by DOI with explicit provenance; Semantic Scholar enrichment requires API access when anonymous requests are rate limited.",
                "Offline collaborator ranking compares random, popularity, TF-IDF, BM25, LSA/SVD, semantic hashing, hybrid and fairness reranking on identical train-only candidates.",
                "The reported exposure gap uses train-period publication count as a low-visibility proxy, not verified career stage.",
                "SciBERT, SPECTER2 and GraphSAGE still need separate evaluation under the same held-out ranking protocol.",
            ],
        )
