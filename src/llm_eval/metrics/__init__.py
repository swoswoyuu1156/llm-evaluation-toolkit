from abc import ABC, abstractmethod

from llm_eval.types import EvalResult


class BaseMetric(ABC):
    """すべての評価指標の基底クラス"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> EvalResult:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


from llm_eval.metrics.bleu import BLEUMetric
from llm_eval.metrics.rouge import ROUGEMetric

__all__ = ["BaseMetric", "BLEUMetric", "ROUGEMetric"]

from llm_eval.metrics.bleu import BLEUMetric
from llm_eval.metrics.rouge import ROUGEMetric
from llm_eval.metrics.semantic import SemanticSimilarityMetric
from llm_eval.metrics.judge import LLMJudgeMetric

__all__ = [
    "BaseMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "SemanticSimilarityMetric",
    "LLMJudgeMetric",
]