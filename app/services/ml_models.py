from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class ModelSimilarityScore:
    name: str
    score: float
    role: str


@dataclass(frozen=True)
class HybridSimilarityScore:
    lexical: float
    bm25: float
    lsa: float
    semantic: float
    hybrid: float
    model_scores: tuple[ModelSimilarityScore, ...]


class HybridSemanticScorer:
    """Comparable local rankers for dissertation experiments and Docker-safe demos."""

    def __init__(
        self,
        lexical_weight: float = 0.23,
        bm25_weight: float = 0.2,
        lsa_weight: float = 0.25,
        semantic_weight: float = 0.32,
    ) -> None:
        self.lexical_weight = lexical_weight
        self.bm25_weight = bm25_weight
        self.lsa_weight = lsa_weight
        self.semantic_weight = semantic_weight
        self.model_name = "TF-IDF + BM25 + LSA/SVD + semantic hashing ensemble"

    def score(self, query_text: str, candidate_texts: list[str]) -> list[HybridSimilarityScore]:
        if not candidate_texts:
            return []

        normalized_query = query_text.strip() or "research profile"
        normalized_candidates = [text.strip() or "unknown scholarly candidate" for text in candidate_texts]
        lexical_scores = self._lexical_scores(normalized_query, normalized_candidates)
        bm25_scores = self._bm25_scores(normalized_query, normalized_candidates)
        lsa_scores = self._lsa_scores(normalized_query, normalized_candidates)
        semantic_scores = self._semantic_scores(normalized_query, normalized_candidates)

        scores: list[HybridSimilarityScore] = []
        for lexical, bm25, lsa, semantic in zip(
            lexical_scores,
            bm25_scores,
            lsa_scores,
            semantic_scores,
            strict=True,
        ):
            hybrid = (
                self.lexical_weight * lexical
                + self.bm25_weight * bm25
                + self.lsa_weight * lsa
                + self.semantic_weight * semantic
            )
            lexical_value = self._clip(lexical)
            bm25_value = self._clip(bm25)
            lsa_value = self._clip(lsa)
            semantic_value = self._clip(semantic)
            hybrid_value = self._clip(hybrid)
            scores.append(
                HybridSimilarityScore(
                    lexical=lexical_value,
                    bm25=bm25_value,
                    lsa=lsa_value,
                    semantic=semantic_value,
                    hybrid=hybrid_value,
                    model_scores=(
                        ModelSimilarityScore("TF-IDF cosine", lexical_value, "lexical baseline"),
                        ModelSimilarityScore("BM25", bm25_value, "information retrieval baseline"),
                        ModelSimilarityScore("LSA/SVD", lsa_value, "latent semantic baseline"),
                        ModelSimilarityScore("Semantic hashing", semantic_value, "embedding fallback"),
                        ModelSimilarityScore("Hybrid ensemble", hybrid_value, "final ranker"),
                    ),
                )
            )
        return scores

    def model_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "name": "TF-IDF cosine",
                "role": "lexical baseline",
                "description": "Compares researcher and candidate texts by weighted n-gram overlap.",
            },
            {
                "name": "BM25",
                "role": "information retrieval baseline",
                "description": "Classic search ranking model that rewards important query terms in candidate text.",
            },
            {
                "name": "LSA/SVD",
                "role": "latent semantic baseline",
                "description": "Projects TF-IDF features into a lower-dimensional latent topic space.",
            },
            {
                "name": "Semantic hashing",
                "role": "embedding fallback",
                "description": "Docker-safe vector representation used until SPECTER2/SciBERT weights are added.",
            },
            {
                "name": "Hybrid ensemble",
                "role": "final ranker",
                "description": "Weighted combination used for the final recommendation order.",
            },
            {
                "name": "SPECTER2/SciBERT",
                "role": "planned transformer embeddings",
                "description": "Pluggable next step for scientific text embeddings; not downloaded in the lightweight demo image.",
            },
        ]

    def _lexical_scores(self, query_text: str, candidate_texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([query_text, *candidate_texts])
        return cosine_similarity(matrix[0:1], matrix[1:])[0]

    def _bm25_scores(self, query_text: str, candidate_texts: list[str]) -> np.ndarray:
        tokenized_candidates = [self._tokens(text) for text in candidate_texts]
        query_terms = self._tokens(query_text)
        if not query_terms or not tokenized_candidates:
            return np.zeros(len(candidate_texts))

        doc_count = len(tokenized_candidates)
        doc_lengths = [len(tokens) or 1 for tokens in tokenized_candidates]
        avg_doc_length = sum(doc_lengths) / max(doc_count, 1)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_candidates:
            document_frequency.update(set(tokens))

        k1 = 1.5
        b = 0.75
        raw_scores: list[float] = []
        for tokens, doc_length in zip(tokenized_candidates, doc_lengths, strict=True):
            term_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * doc_length / avg_doc_length)
                score += idf * (frequency * (k1 + 1)) / denominator
            raw_scores.append(score)
        return self._normalize_scores(raw_scores)

    def _lsa_scores(self, query_text: str, candidate_texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([query_text, *candidate_texts])
        max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1, 64)
        if max_components < 1:
            return self._lexical_scores(query_text, candidate_texts)

        svd = TruncatedSVD(n_components=max_components, random_state=42)
        latent = normalize(svd.fit_transform(matrix))
        return cosine_similarity(latent[0:1], latent[1:])[0]

    def _semantic_scores(self, query_text: str, candidate_texts: list[str]) -> np.ndarray:
        vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 3),
            stop_words="english",
        )
        matrix = vectorizer.transform([query_text, *candidate_texts])
        return cosine_similarity(matrix[0:1], matrix[1:])[0]

    def _tokens(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]

    def _normalize_scores(self, scores: list[float]) -> np.ndarray:
        if not scores:
            return np.array([])
        maximum = max(scores)
        if maximum <= 0:
            return np.zeros(len(scores))
        return np.array([score / maximum for score in scores])

    def _clip(self, value: float) -> float:
        return round(float(np.clip(value, 0.0, 1.0)), 4)
