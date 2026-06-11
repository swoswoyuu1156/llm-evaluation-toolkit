from __future__ import annotations

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