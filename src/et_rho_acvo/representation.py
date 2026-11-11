from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from et_rho_acvo.text import tokenize_documents, tokenize_text


class EnsembleTextRepresentation:
    """Fit and apply the TF-IDF/SVD + Word2Vec + LDA ensemble."""

    def __init__(self, config: dict[str, Any], word2vec_params: dict[str, Any], seed: int):
        text_cfg = config.get("text", {})
        rep_cfg = config.get("representation", {})
        self.language = str(text_cfg.get("language", "en"))
        self.min_token_length = int(text_cfg.get("min_token_length", 1))
        tokenizer = partial(tokenize_text, language=self.language, min_token_length=self.min_token_length)
        self.tfidf = TfidfVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=int(rep_cfg.get("tfidf_max_features", 20000)),
            min_df=int(rep_cfg.get("tfidf_min_df", 1)),
        )
        self.count = CountVectorizer(
            tokenizer=tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=int(rep_cfg.get("tfidf_max_features", 20000)),
            min_df=int(rep_cfg.get("tfidf_min_df", 1)),
        )
        self.svd_components = int(rep_cfg.get("svd_components", 100))
        self.lda_topics = int(rep_cfg.get("lda_topics", 20))
        self.lda_max_iter = int(rep_cfg.get("lda_max_iter", 20))
        self.w2v_epochs = int(rep_cfg.get("word2vec_epochs", 10))
        self.w2v_min_count = int(rep_cfg.get("word2vec_min_count", 1))
        self.w2v_workers = int(rep_cfg.get("word2vec_workers", 1))
        self.word2vec_params = {
            "vector_size": int(word2vec_params.get("vector_size", rep_cfg.get("word2vec_vector_size_default", 100))),
            "window": int(word2vec_params.get("window", rep_cfg.get("word2vec_window_default", 5))),
            "sg": int(word2vec_params.get("sg", rep_cfg.get("word2vec_sg_default", 1))),
        }
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.word_vectors: Word2Vec | None = None
        self.svd: TruncatedSVD | None = None
        self.lda: LatentDirichletAllocation | None = None
        self._fitted = False

    def _word2vec_features(self, tokenized: list[list[str]]) -> np.ndarray:
        if self.word_vectors is None:
            raise RuntimeError("Word2Vec has not been fitted.")
        width = self.word2vec_params["vector_size"]
        result = np.zeros((len(tokenized), width), dtype=np.float32)
        keyed = self.word_vectors.wv
        for row, tokens in enumerate(tokenized):
            vectors = [keyed[token] for token in tokens if token in keyed]
            if vectors:
                result[row] = np.mean(vectors, axis=0)
        return result

    def fit(self, documents: list[str]) -> "EnsembleTextRepresentation":
        docs = [str(doc) for doc in documents]
        tokenized = tokenize_documents(docs, self.language, self.min_token_length)
        if not any(tokenized):
            raise ValueError("The training documents contain no tokens after tokenization.")

        tfidf_matrix = self.tfidf.fit_transform(docs)
        if tfidf_matrix.shape[1] > 1 and tfidf_matrix.shape[0] > 1:
            n_components = min(self.svd_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
            n_components = max(1, n_components)
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
            tfidf_features = self.svd.fit_transform(tfidf_matrix)
        else:
            self.svd = None
            tfidf_features = tfidf_matrix.toarray()

        self.word_vectors = Word2Vec(
            sentences=tokenized,
            vector_size=self.word2vec_params["vector_size"],
            window=self.word2vec_params["window"],
            sg=self.word2vec_params["sg"],
            min_count=self.w2v_min_count,
            workers=self.w2v_workers,
            seed=self.seed,
            epochs=self.w2v_epochs,
        )
        word_features = self._word2vec_features(tokenized)

        count_matrix = self.count.fit_transform(docs)
        self.lda = LatentDirichletAllocation(
            n_components=self.lda_topics,
            learning_method="batch",
            max_iter=self.lda_max_iter,
            random_state=self.seed,
        )
        lda_features = self.lda.fit_transform(count_matrix)

        combined = np.hstack([tfidf_features, word_features, lda_features]).astype(np.float32)
        self.scaler.fit(combined)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit before transform.")
        docs = [str(doc) for doc in documents]
        tokenized = tokenize_documents(docs, self.language, self.min_token_length)
        tfidf_matrix = self.tfidf.transform(docs)
        if self.svd is None:
            tfidf_features = tfidf_matrix.toarray()
        else:
            tfidf_features = self.svd.transform(tfidf_matrix)
        word_features = self._word2vec_features(tokenized)
        count_matrix = self.count.transform(docs)
        lda_features = self.lda.transform(count_matrix)
        combined = np.hstack([tfidf_features, word_features, lda_features]).astype(np.float32)
        return self.scaler.transform(combined).astype(np.float32)
