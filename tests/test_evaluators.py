from llm_eval.evaluators import BaseEvaluator
from llm_eval.metrics import BLEUMetric, ROUGEMetric
from llm_eval.types import EvalResult


class ConcreteEvaluator(BaseEvaluator):
    """テスト用の具体的なEvaluator"""
    pass


class TestBaseEvaluator:
    def test_evaluate_returns_dict(
        self, sample_predictions, sample_references
    ):
        evaluator = ConcreteEvaluator(
            metrics=[BLEUMetric(), ROUGEMetric()]
        )
        results = evaluator.evaluate(sample_predictions, sample_references)
        assert isinstance(results, dict)
        assert "bleu" in results
        assert "rouge" in results

    def test_all_results_are_eval_result(
        self, sample_predictions, sample_references
    ):
        evaluator = ConcreteEvaluator(metrics=[BLEUMetric()])
        results = evaluator.evaluate(sample_predictions, sample_references)
        for result in results.values():
            assert isinstance(result, EvalResult)

    def test_add_metric(self, sample_predictions, sample_references):
        evaluator = ConcreteEvaluator(metrics=[BLEUMetric()])
        evaluator.add_metric(ROUGEMetric())
        assert len(evaluator.metrics) == 2

    def test_repr(self):
        evaluator = ConcreteEvaluator(metrics=[BLEUMetric(), ROUGEMetric()])
        assert "bleu" in repr(evaluator)
        assert "rouge" in repr(evaluator)