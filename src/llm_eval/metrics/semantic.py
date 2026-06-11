from __future__ import annotations

import numpy as np

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult


class SemanticSimilarityMetric(BaseMetric):
    """
    埋め込みベクトルのコサイン類似度で意味的な近さを評価する

    BLEUやROUGEでは捉えられない言い換えや同義語を考慮できる。
    スコアは0.0〜1.0で、1.0が意味的に完全一致。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name="semantic_similarity")
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """モデルの遅延読み込み（初回呼び出し時にダウンロード）"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformersが必要です: "
                    "pip install llm-evaluation-toolkit[semantic]"
                )
            print(f"モデルを読み込み中: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """2つのベクトルのコサイン類似度を計算する"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> EvalResult:
        if len(predictions) != len(references):
            raise ValueError(
                f"predictionsとreferencesの数が一致しません: "
                f"{len(predictions)} != {len(references)}"
            )

        model = self._load_model()

        pred_embeddings = model.encode(predictions, convert_to_numpy=True)
        ref_embeddings = model.encode(references, convert_to_numpy=True)

        similarities = [
            self._cosine_similarity(pred_embeddings[i], ref_embeddings[i])
            for i in range(len(predictions))
        ]

        avg_score = float(np.mean(similarities))

        return EvalResult(
            metric_name=self.name,
            score=avg_score,
            details={
                "num_samples": len(predictions),
                "model": self.model_name,
                "individual_scores": similarities,
                "min_score": float(np.min(similarities)),
                "max_score": float(np.max(similarities)),
            },
        )