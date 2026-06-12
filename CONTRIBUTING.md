# llm-evaluation-toolkit への貢献ガイド
# Contributing to llm-evaluation-toolkit

貢献に興味を持っていただきありがとうございます！

*Thank you for your interest in contributing!*

---

## はじめに / Getting Started

1. リポジトリをフォークする / Fork the repository
2. `develop` ブランチからフィーチャーブランチを作成する / Create a feature branch from `develop`
```bash
   git checkout develop
   git checkout -b feat/your-feature-name
```
3. 変更を加える / Make your changes
4. テストとリントを実行する / Run tests and linter
```bash
   ruff check src/ tests/
   pytest tests/ -v --cov=src/llm_eval
```
5. コミットしてプッシュする / Commit and push
```bash
   git commit -m "feat: your feature description"
   git push origin feat/your-feature-name
```
6. `develop` ブランチへのPull Requestを作成する / Open a Pull Request to `develop`

---

## 新しい評価指標を追加する / Adding a New Metric

1. `src/llm_eval/metrics/your_metric.py` を作成する / Create `src/llm_eval/metrics/your_metric.py`
2. `BaseMetric` を継承して `compute()` を実装する / Inherit from `BaseMetric` and implement `compute()`
3. `src/llm_eval/metrics/__init__.py` からエクスポートする / Export from `src/llm_eval/metrics/__init__.py`
4. `tests/test_metrics.py` にテストを追加する / Add tests in `tests/test_metrics.py`

```python
from __future__ import annotations

from llm_eval.metrics.base import BaseMetric
from llm_eval.types import EvalResult


class YourMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="your_metric")

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> EvalResult:
        # ここに実装を書く / your implementation here
        return EvalResult(metric_name=self.name, score=0.0)
```

---

## コミットメッセージの形式 / Commit Message Format

| プレフィックス / Prefix | 用途 / Usage |
|------------------------|-------------|
| `feat:` | 新機能 / New feature |
| `fix:` | バグ修正 / Bug fix |
| `test:` | テスト追加 / Adding tests |
| `docs:` | ドキュメント更新 / Documentation |
| `ci:` | CI/CD関連 / CI/CD changes |
| `chore:` | メンテナンス / Maintenance |

---

## コードスタイル / Code Style

このプロジェクトは `ruff` を使用しています。コミット前に必ず実行してください。

*This project uses `ruff` for linting. Run before committing:*

```bash
ruff check src/ tests/
```

---

## 質問・提案 / Questions & Suggestions

バグ報告や機能提案は [Issues](https://github.com/swoswoyuu1156/llm-evaluation-toolkit/issues) からお気軽にどうぞ。

*For bug reports and feature requests, feel free to open an [Issue](https://github.com/swoswoyuu1156/llm-evaluation-toolkit/issues).*