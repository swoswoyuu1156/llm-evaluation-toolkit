from __future__ import annotations

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult


class BLEUMetric(BaseMetric):
    """
    BLEU (Bilingual Evaluation Understudy) スコアを計算する

    機械翻訳や要約タスクでLLM出力と正解文の一致度を測る。
    スコアは0.0〜1.0で、1.0が完全一致。
    """

    def __init__(self, max_n: int = 4):
        super().__init__(name="bleu")
        self.max_n = max_n
        self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

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

        tokenized_predictions = [p.lower().split() for p in predictions]
        tokenized_references = [[r.lower().split()] for r in references]

        smoothing = SmoothingFunction().method1
        score = corpus_bleu(
            tokenized_references,
            tokenized_predictions,
            smoothing_function=smoothing,
        )

        return EvalResult(
            metric_name=self.name,
            score=float(score),
            details={
                "num_samples": len(predictions),
                "max_n": self.max_n,
            },
        )