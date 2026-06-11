import pytest
from llm_eval.types import EvalResult
from llm_eval.datasets import DatasetLoader, EvalDataset


class TestEvalResult:
    def test_basic_creation(self):
        result = EvalResult(metric_name="test", score=0.85)
        assert result.metric_name == "test"
        assert result.score == 0.85
        assert result.details == {}

    def test_with_details(self):
        result = EvalResult(
            metric_name="bleu",
            score=0.5,
            details={"num_samples": 10},
        )
        assert result.details["num_samples"] == 10

    def test_repr(self):
        result = EvalResult(metric_name="bleu", score=0.1234)
        assert "bleu" in repr(result)
        assert "0.1234" in repr(result)


class TestDatasetLoader:
    def test_from_dict(self):
        dataset = DatasetLoader.from_dict(
            name="test",
            questions=["Q1", "Q2"],
            references=["A1", "A2"],
        )
        assert len(dataset) == 2
        assert dataset.name == "test"

    def test_subset(self):
        dataset = DatasetLoader.from_dict(
            name="test",
            questions=["Q1", "Q2", "Q3"],
            references=["A1", "A2", "A3"],
        )
        small = dataset.subset(2)
        assert len(small) == 2
        assert small.questions == ["Q1", "Q2"]

    def test_mismatched_lengths_raise_error(self):
        with pytest.raises(ValueError):
            EvalDataset(
                name="bad",
                questions=["Q1", "Q2"],
                references=["A1"],
            )