from __future__ import annotations

from abc import ABC

from llm_eval.metrics import BaseMetric
from llm_eval.types import EvalResult


class BaseEvaluator(ABC):
    """複数のMetricをまとめて実行する基底クラス"""

    def __init__(self, metrics: list[BaseMetric]):
        self.metrics = metrics

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, EvalResult]:
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute(predictions, references)
        return results

    def add_metric(self, metric: BaseMetric) -> None:
        self.metrics.append(metric)

    def __repr__(self) -> str:
        metric_names = [m.name for m in self.metrics]
        return f"{self.__class__.__name__}(metrics={metric_names})"