from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """評価結果を格納するデータクラス"""

    metric_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"EvalResult(metric={self.metric_name}, score={self.score:.4f})"