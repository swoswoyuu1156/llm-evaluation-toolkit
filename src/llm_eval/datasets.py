from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalDataset:
    """評価用データセットを格納するクラス"""
    name: str
    questions: list[str]
    references: list[str]
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if len(self.questions) != len(self.references):
            raise ValueError(
                f"questionsとreferencesの数が一致しません: "
                f"{len(self.questions)} != {len(self.references)}"
            )

    def __len__(self) -> int:
        return len(self.questions)

    def __repr__(self) -> str:
        return f"EvalDataset(name={self.name}, size={len(self)})"

    def subset(self, n: int) -> "EvalDataset":
        """先頭n件だけ取り出す"""
        return EvalDataset(
            name=f"{self.name}(subset={n})",
            questions=self.questions[:n],
            references=self.references[:n],
            metadata=self.metadata,
        )


class DatasetLoader:
    """HuggingFace datasetsからベンチマークを読み込むクラス"""

    @staticmethod
    def _import_datasets():
        try:
            import datasets
            return datasets
        except ImportError:
            raise ImportError(
                "datasetsパッケージが必要です: pip install datasets"
            )

    @classmethod
    def load_squad(cls, split: str = "validation", max_samples: int = 100) -> EvalDataset:
        """
        SQuAD (Stanford Question Answering Dataset) を読み込む

        質問応答タスクの定番ベンチマーク。
        questions: 質問文
        references: 正解テキスト
        """
        ds = cls._import_datasets()
        dataset = ds.load_dataset("squad", split=split)

        questions = []
        references = []

        for item in dataset.select(range(min(max_samples, len(dataset)))):
            questions.append(item["question"])
            references.append(item["answers"]["text"][0])

        return EvalDataset(
            name="squad",
            questions=questions,
            references=references,
            metadata={"split": split, "max_samples": max_samples},
        )

    @classmethod
    def load_cnn_dailymail(cls, split: str = "validation", max_samples: int = 50) -> EvalDataset:
        """
        CNN/DailyMail を読み込む

        要約タスクの定番ベンチマーク。
        questions: 元記事
        references: 正解要約
        """
        ds = cls._import_datasets()
        dataset = ds.load_dataset("cnn_dailymail", "3.0.0", split=split)

        questions = []
        references = []

        for item in dataset.select(range(min(max_samples, len(dataset)))):
            questions.append(item["article"][:512])
            references.append(item["highlights"])

        return EvalDataset(
            name="cnn_dailymail",
            questions=questions,
            references=references,
            metadata={"split": split, "max_samples": max_samples},
        )

    @classmethod
    def from_dict(
        cls,
        name: str,
        questions: list[str],
        references: list[str],
    ) -> EvalDataset:
        """
        自前のデータからEvalDatasetを作る

        APIキーなしでテストしたいときに便利。
        """
        return EvalDataset(
            name=name,
            questions=questions,
            references=references,
        )