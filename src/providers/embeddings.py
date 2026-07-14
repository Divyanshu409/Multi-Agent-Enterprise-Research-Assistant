from __future__ import annotations

import hashlib

import numpy as np

from src.providers.base import EmbeddingProvider

from dotenv import load_dotenv

load_dotenv()


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed. Run "
                "`pip install sentence-transformers`."
            ) from e
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str, api_key: str | None = None):
        import os

        try:
            import openai
        except ImportError as e:  
            raise RuntimeError("openai not installed. Run `pip install openai`.") from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self._client = openai.OpenAI(api_key=key)
        self._model_name = model_name
        self._dim = 1536  

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"


class LocalTfidfEmbedding(EmbeddingProvider):
    _DIM = 384 

    def __init__(self, vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer: TfidfVectorizer | None = vectorizer

    def fit(self, corpus_texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=8192, ngram_range=(1, 2)
        )
        self._vectorizer.fit(corpus_texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._vectorizer is None:
            raise RuntimeError(
                "LocalTfidfEmbedding must be fit (or loaded from a saved fit) "
                "before embed() -- run build_index.py first."
            )
        sparse = self._vectorizer.transform(texts)  
        dense = self._hash_project(sparse)

        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).tolist()

    def _hash_project(self, sparse_matrix) -> np.ndarray:
        n_docs = sparse_matrix.shape[0]
        out = np.zeros((n_docs, self._DIM), dtype=np.float32)
        coo = sparse_matrix.tocoo()
        for row, col, value in zip(coo.row, coo.col, coo.data):
            bucket = int(hashlib.md5(str(col).encode()).hexdigest(), 16) % self._DIM
            out[row, bucket] += value
        return out

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self._vectorizer, f)

    @classmethod
    def load(cls, path: str) -> "LocalTfidfEmbedding":
        import pickle

        with open(path, "rb") as f:
            vectorizer = pickle.load(f)
        return cls(vectorizer=vectorizer)

    @property
    def dimension(self) -> int:
        return self._DIM

    @property
    def name(self) -> str:
        return "local-tfidf"


def get_embedding_provider(provider_name: str, **kwargs) -> EmbeddingProvider:
    if provider_name == "sentence-transformers":
        return SentenceTransformerEmbedding(kwargs["model_name"])
    if provider_name == "openai":
        return OpenAIEmbedding(kwargs["model_name"], api_key=kwargs.get("api_key"))
    if provider_name == "local-tfidf":
        vectorizer_path = kwargs.get("vectorizer_path")
        if vectorizer_path:
            return LocalTfidfEmbedding.load(vectorizer_path)
        return LocalTfidfEmbedding()
    raise ValueError(f"Unknown embedding provider: {provider_name}")
