# llm-evaluation-toolkit

[![CI](https://github.com/swoswoyuu1156/llm-evaluation-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/swoswoyuu1156/llm-evaluation-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/llm-evaluation-toolkit.svg)](https://badge.fury.io/py/llm-evaluation-toolkit)

LLMの出力を評価するための軽量Pythonライブラリです。BLEU・ROUGE・意味的類似度・LLM-as-a-Judgeの4種類の評価指標を、統一されたAPIで利用できます。

*A lightweight Python toolkit for evaluating LLM outputs. Supports BLEU, ROUGE, semantic similarity, and LLM-as-a-Judge — all with a unified API.*

---

## なぜ llm-evaluation-toolkit なのか？ / Why llm-evaluation-toolkit?

既存の評価ライブラリは研究用途に特化していたり、セットアップが複雑すぎる問題がありました。このライブラリは、実際の開発ワークフローでLLMの出力を素早く評価したい開発者のために設計されています。

*Existing evaluation libraries are either too research-focused or require complex setup. This toolkit is designed for developers who need to evaluate LLM outputs quickly in real workflows.*

- **統一API** — すべての評価指標が同じインターフェースを持つ / *Unified API — all metrics share the same interface*
- **複数プロバイダ対応** — OpenAI・Anthropicをすぐに利用可能 / *Multiple providers — OpenAI and Anthropic out of the box*
- **正解テキスト不要** — LLM-as-a-Judgeは参照テキストなしで評価可能 / *No reference needed — LLM-as-a-Judge works without ground truth*
- **軽量設計** — 必要な機能だけインストールできる / *Lightweight — install only what you need*

---

## インストール / Installation

```bash
# 基本インストール / Base installation
pip install llm-evaluation-toolkit

# OpenAI対応 / With OpenAI support
pip install llm-evaluation-toolkit[openai]

# Anthropic対応 / With Anthropic support
pip install llm-evaluation-toolkit[anthropic]

# 意味的類似度対応 / With semantic similarity support
pip install llm-evaluation-toolkit[semantic]

# 全機能 / All features
pip install llm-evaluation-toolkit[openai,anthropic,semantic]
```

---

## クイックスタート / Quick Start

### 基本的な評価 / Basic evaluation

```python
from llm_eval import BLEUMetric, ROUGEMetric, SemanticSimilarityMetric

predictions = [
    "The cat is on the mat",
    "A dog was running in the park",
]
references = [
    "A cat is sitting on a mat",
    "The dog ran through the park",
]

bleu = BLEUMetric()
rouge = ROUGEMetric()
semantic = SemanticSimilarityMetric()

print(bleu.compute(predictions, references))
# EvalResult(metric=bleu, score=0.4231)

print(rouge.compute(predictions, references))
# EvalResult(metric=rouge, score=0.6842)

print(semantic.compute(predictions, references))
# EvalResult(metric=semantic_similarity, score=0.8923)
```

### 正解テキストなしで評価（LLM-as-a-Judge） / Evaluate without reference texts

```python
from llm_eval import LLMJudgeMetric

# 正解テキストなしで出力品質を評価できる
# Evaluate output quality without any reference text
judge = LLMJudgeMetric(judge_model="gpt-4o-mini")

questions = ["日本の首都はどこですか？"]
answers = ["日本の首都は東京です。政治・経済・文化の中心地として機能しています。"]

result = judge.compute(answers, questions)
print(result)
# EvalResult(metric=llm_judge, score=0.9)
```

### 複数の評価指標を一括実行 / Run multiple metrics at once

```python
from llm_eval import BaseEvaluator, BLEUMetric, ROUGEMetric, SemanticSimilarityMetric

evaluator = BaseEvaluator(metrics=[
    BLEUMetric(),
    ROUGEMetric(),
    SemanticSimilarityMetric(),
])

results = evaluator.evaluate(predictions, references)
for metric_name, result in results.items():
    print(f"{metric_name}: {result.score:.4f}")
```

### LLMプロバイダとの連携 / Use with LLM providers

```python
from llm_eval import OpenAIProvider, GenerationConfig, BLEUMetric

# LLMで回答を生成してそのまま評価する
# Generate predictions from LLM and evaluate directly
provider = OpenAIProvider(
    model="gpt-4o-mini",
    config=GenerationConfig(temperature=0.0, max_tokens=256),
)

questions = ["機械学習とは何ですか？", "ニューラルネットワークを説明してください。"]
predictions = provider.generate_batch(questions)

references = [
    "機械学習はデータから学習するAIの一分野です。",
    "ニューラルネットワークは人間の脳を模倣した計算システムです。",
]

result = BLEUMetric().compute(predictions, references)
print(result)
```

### ベンチマークデータセットの利用 / Load benchmark datasets

```python
from llm_eval import DatasetLoader

# 組み込みベンチマークを使う / Use built-in benchmarks
squad = DatasetLoader.load_squad(max_samples=50)
print(squad)
# EvalDataset(name=squad, size=50)

# 自前データを使う / Use your own data
dataset = DatasetLoader.from_dict(
    name="my_dataset",
    questions=["質問1", "質問2"],
    references=["回答1", "回答2"],
)
```

---

## 対応評価指標 / Supported Metrics

| 指標 / Metric | 正解テキスト / Reference | 適したタスク / Best for |
|--------------|------------------------|------------------------|
| BLEU | 必要 / Yes | 翻訳・固定フォーマット生成 / Translation, fixed-format generation |
| ROUGE | 必要 / Yes | 要約 / Summarization |
| Semantic Similarity | 必要 / Yes | 言い換えを含むタスク / Paraphrase-heavy tasks |
| LLM-as-a-Judge | 不要 / No | 自由記述生成 / Open-ended generation |

---

## 対応プロバイダ / Supported Providers

| プロバイダ / Provider | インストール / Install | モデル例 / Models |
|----------------------|----------------------|------------------|
| OpenAI | `pip install llm-evaluation-toolkit[openai]` | gpt-4o, gpt-4o-mini |
| Anthropic | `pip install llm-evaluation-toolkit[anthropic]` | claude-opus-4-6, claude-haiku-4-5 |

---

## 環境変数 / Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## 開発環境のセットアップ / Development Setup

```bash
git clone https://github.com/swoswoyuu1156/llm-evaluation-toolkit.git
cd llm-evaluation-toolkit
python -m venv .venv

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e ".[dev]"

# テスト実行 / Run tests
pytest tests/ -v --cov=src/llm_eval

# リント実行 / Run linter
ruff check src/ tests/
```

---

## コントリビューション / Contributing

コントリビューションを歓迎します！まず [CONTRIBUTING.md](CONTRIBUTING.md) をご確認ください。

*Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.*

---

## ライセンス / License

MIT License — 詳細は [LICENSE](LICENSE) をご確認ください。

*MIT License — see [LICENSE](LICENSE) for details.*