from rouge_score import rouge_scorer

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult


class ROUGEMetric(BaseMetric):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) スコアを計算する

    要約タスクでよく使われる。
    ROUGE-1: 単語の一致率
    ROUGE-2: 2単語フレーズの一致率
    ROUGE-L: 最長共通部分列ベースの一致率
    """

    def __init__(self, rouge_types: list[str] | None = None):
        super().__init__(name="rouge")
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]
        self._scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=True,
        )

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

        aggregated: dict[str, list[float]] = {t: [] for t in self.rouge_types}

        for pred, ref in zip(predictions, references):
            scores = self._scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                aggregated[rouge_type].append(scores[rouge_type].fmeasure)

        details = {}
        for rouge_type, values in aggregated.items():
            details[rouge_type] = sum(values) / len(values)

        primary_score = details.get("rougeL", details[self.rouge_types[0]])

        return EvalResult(
            metric_name=self.name,
            score=primary_score,
            details=details,
        )