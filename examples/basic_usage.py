"""
llm-evaluation-toolkit の基本的な使い方
"""
from llm_eval import (
    BLEUMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
    BaseEvaluator,
    DatasetLoader,
)


def example_basic_metrics():
    """基本的なメトリクスの使い方"""
    print("=== Basic Metrics ===")

    predictions = [
        "The cat is on the mat",
        "A dog was running in the park",
        "The sun is shining brightly today",
    ]
    references = [
        "A cat is sitting on a mat",
        "The dog ran through the park",
        "It is a bright and sunny day",
    ]

    for metric in [BLEUMetric(), ROUGEMetric(), SemanticSimilarityMetric()]:
        result = metric.compute(predictions, references)
        print(f"{result.metric_name:25s}: {result.score:.4f}")


def example_evaluator():
    """複数メトリクスを一括実行する"""
    print("\n=== Evaluator (multiple metrics) ===")

    predictions = ["Tokyo is the capital of Japan."]
    references = ["Japan's capital city is Tokyo."]

    evaluator = BaseEvaluator(metrics=[
        BLEUMetric(),
        ROUGEMetric(),
        SemanticSimilarityMetric(),
    ])

    results = evaluator.evaluate(predictions, references)
    for name, result in results.items():
        print(f"{name:25s}: {result.score:.4f}")


def example_dataset():
    """データセットの使い方"""
    print("\n=== Dataset Loader ===")

    dataset = DatasetLoader.from_dict(
        name="example",
        questions=["What is AI?", "What is Python?"],
        references=[
            "AI is artificial intelligence.",
            "Python is a programming language.",
        ],
    )

    print(f"Dataset: {dataset}")
    print(f"Size: {len(dataset)}")
    print(f"First question: {dataset.questions[0]}")


if __name__ == "__main__":
    example_basic_metrics()
    example_evaluator()
    example_dataset()