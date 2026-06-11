from __future__ import annotations

import os
import re

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for AI-generated text.

Evaluate the following response based on these criteria:
- Accuracy: Is the information correct?
- Clarity: Is the response clear and easy to understand?
- Completeness: Does it fully address the question?
- Relevance: Is it relevant to the question asked?

Question: {question}
Response: {response}

Provide your evaluation in the following format:
Score: [0-10]
Reasoning: [brief explanation]

Be strict but fair. A score of 10 means perfect."""


class LLMJudgeMetric(BaseMetric):
    """
    LLM自身が審判として出力を採点するメトリクス

    正解テキストなしで出力の品質を評価できる。
    スコアは0.0〜1.0に正規化される（元スコアは0〜10）。
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        prompt_template: str | None = None,
    ):
        super().__init__(name="llm_judge")
        self.judge_model = judge_model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.prompt_template = prompt_template or JUDGE_PROMPT_TEMPLATE
        self._client = None

    def _get_client(self):
        """OpenAIクライアントの遅延初期化"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openaiパッケージが必要です: "
                    "pip install llm-evaluation-toolkit[openai]"
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _parse_score(self, judge_response: str) -> float:
        """LLMの採点レスポンスからスコアを抽出する"""
        pattern = r"Score:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, judge_response, re.IGNORECASE)
        if match:
            raw_score = float(match.group(1))
            return min(max(raw_score / 10.0, 0.0), 1.0)
        return 0.5

    def _judge_single(self, question: str, response: str) -> tuple[float, str]:
        """1つのレスポンスを採点する"""
        client = self._get_client()
        prompt = self.prompt_template.format(
            question=question,
            response=response,
        )
        result = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        judge_response = result.choices[0].message.content or ""
        score = self._parse_score(judge_response)
        return score, judge_response

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> EvalResult:
        """
        Args:
            predictions: LLMの出力テキストのリスト
            references:  対応する質問文のリスト（このMetricでは正解ではなく質問を渡す）
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictionsとreferencesの数が一致しません: "
                f"{len(predictions)} != {len(references)}"
            )

        scores = []
        judge_responses = []

        for question, response in zip(references, predictions):
            score, judge_response = self._judge_single(question, response)
            scores.append(score)
            judge_responses.append(judge_response)
            print(f"採点完了: {score:.2f} / 1.0")

        avg_score = sum(scores) / len(scores)

        return EvalResult(
            metric_name=self.name,
            score=avg_score,
            details={
                "num_samples": len(predictions),
                "judge_model": self.judge_model,
                "individual_scores": scores,
                "judge_responses": judge_responses,
            },
        )