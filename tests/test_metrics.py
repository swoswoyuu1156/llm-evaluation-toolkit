import pytest
from unittest.mock import MagicMock, patch
from llm_eval.metrics import BLEUMetric, ROUGEMetric, SemanticSimilarityMetric
from llm_eval.metrics.judge import LLMJudgeMetric
from llm_eval.types import EvalResult


class TestBLEUMetric:
    def test_perfect_match(self):
        metric = BLEUMetric()
        texts = ["the cat sat on the mat"]
        result = metric.compute(texts, texts)
        assert result.score > 0.9
        assert result.metric_name == "bleu"

    def test_no_match(self):
        metric = BLEUMetric()
        predictions = ["hello world foo bar"]
        references = ["completely different text here"]
        result = metric.compute(predictions, references)
        assert 0.0 <= result.score <= 1.0

    def test_returns_eval_result(self):
        metric = BLEUMetric()
        result = metric.compute(["hello"], ["hello"])
        assert isinstance(result, EvalResult)

    def test_mismatched_lengths_raise_error(self):
        metric = BLEUMetric()
        with pytest.raises(ValueError):
            metric.compute(["a", "b"], ["a"])


class TestROUGEMetric:
    def test_perfect_match(self):
        metric = ROUGEMetric()
        texts = ["the cat sat on the mat"]
        result = metric.compute(texts, texts)
        assert result.score > 0.9

    def test_details_contain_rouge_types(self):
        metric = ROUGEMetric()
        result = metric.compute(["hello world"], ["hello world"])
        assert "rouge1" in result.details
        assert "rouge2" in result.details
        assert "rougeL" in result.details

    def test_returns_eval_result(self):
        metric = ROUGEMetric()
        result = metric.compute(["hello"], ["hello"])
        assert isinstance(result, EvalResult)

    def test_mismatched_lengths_raise_error(self):
        metric = ROUGEMetric()
        with pytest.raises(ValueError):
            metric.compute(["a", "b"], ["a"])


class TestSemanticSimilarityMetric:
    def _make_mock_metric(self):
        """モック済みのSemanticSimilarityMetricを返す"""
        import numpy as np
        from unittest.mock import MagicMock

        metric = SemanticSimilarityMetric()

        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kwargs: np.array([
            [1.0, 0.0, 0.0] if i % 2 == 0 else [0.9, 0.1, 0.0]
            for i in range(len(texts))
        ])
        metric._model = mock_model
        return metric

    def test_high_similarity_for_similar_texts(
        self, sample_predictions, sample_references
    ):
        metric = self._make_mock_metric()
        result = metric.compute(sample_predictions, sample_references)
        assert result.score > 0.5

    def test_perfect_match_score(self):
        import numpy as np
        from unittest.mock import MagicMock

        metric = SemanticSimilarityMetric()
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kwargs: np.array([
            [1.0, 0.0, 0.0] for _ in texts
        ])
        metric._model = mock_model

        texts = ["the cat sat on the mat"]
        result = metric.compute(texts, texts)
        assert result.score > 0.99

    def test_details_contain_individual_scores(
        self, sample_predictions, sample_references
    ):
        metric = self._make_mock_metric()
        result = metric.compute(sample_predictions, sample_references)
        assert "individual_scores" in result.details
        assert len(result.details["individual_scores"]) == len(sample_predictions)

    def test_mismatched_lengths_raise_error(self):
        metric = SemanticSimilarityMetric()
        with pytest.raises(ValueError):
            metric.compute(["a", "b"], ["a"])


class TestLLMJudgeMetric:
    def test_parse_score(self):
        metric = LLMJudgeMetric()
        assert metric._parse_score("Score: 8\nReasoning: Good.") == 0.8
        assert metric._parse_score("Score: 10\nReasoning: Perfect.") == 1.0
        assert metric._parse_score("Score: 0\nReasoning: Bad.") == 0.0

    def test_parse_score_fallback(self):
        metric = LLMJudgeMetric()
        score = metric._parse_score("no score here")
        assert score == 0.5

    @patch("llm_eval.metrics.judge.LLMJudgeMetric._get_client")
    def test_compute_with_mock(self, mock_get_client, sample_questions):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Score: 8\nReasoning: Good response."
        mock_client.chat.completions.create.return_value = mock_response

        metric = LLMJudgeMetric()
        predictions = ["Tokyo is the capital of Japan."] * len(sample_questions)
        result = metric.compute(predictions, sample_questions)

        assert isinstance(result, EvalResult)
        assert abs(result.score - 0.8) < 1e-9
        assert result.metric_name == "llm_judge"