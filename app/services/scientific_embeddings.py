from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


MODEL_ALIASES = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "specter2": "allenai/specter2_base",
}


class ScientificEmbeddingModel:
    def __init__(self, model_name: str = "hashing", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.vector_size = 768
        self._transformer = None
        self._tokenizer = None
        self._torch = None
        if model_name != "hashing":
            self._load_transformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "hashing":
            return self._hashing_embeddings(texts)
        return self._transformer_embeddings(texts)

    def text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hashing_embeddings(self, texts: list[str]) -> np.ndarray:
        vectorizer = HashingVectorizer(
            n_features=self.vector_size,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 3),
            stop_words="english",
        )
        matrix = vectorizer.transform(texts)
        return normalize(matrix).toarray().astype("float32")

    def _load_transformer(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer embeddings require optional dependencies. "
                "Install them with: pip install -r requirements-ml.txt"
            ) from exc

        resolved = MODEL_ALIASES.get(model_name, model_name)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(resolved)
        self._transformer = AutoModel.from_pretrained(resolved)
        self._transformer.eval()

    def _transformer_embeddings(self, texts: list[str]) -> np.ndarray:
        if not self._torch or not self._tokenizer or not self._transformer:
            raise RuntimeError("Transformer model was not loaded")

        embeddings: list[np.ndarray] = []
        with self._torch.no_grad():
            for batch in self._batches(texts, self.batch_size):
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self._transformer(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu().numpy().astype("float32"))
        return np.vstack(embeddings)

    def _batches(self, items: list[str], size: int) -> Iterable[list[str]]:
        for index in range(0, len(items), size):
            yield items[index : index + size]
