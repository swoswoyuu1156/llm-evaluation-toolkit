from abc import ABC, abstractmethod
from typing import Union

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
        """
        評価スコアを計算する

        Args:
            predictions: LLMの出力テキストのリスト
            references:  正解テキストのリスト

        Returns:
            EvalResult: スコアと詳細情報
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"