# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2024-01-01

### Added
- `BLEUMetric` for translation and generation evaluation
- `ROUGEMetric` for summarization evaluation  
- `SemanticSimilarityMetric` using sentence-transformers
- `LLMJudgeMetric` for reference-free evaluation
- `OpenAIProvider` and `AnthropicProvider` for LLM integration
- `DatasetLoader` with SQuAD and CNN/DailyMail support
- `BaseEvaluator` for running multiple metrics at once
- GitHub Actions CI/CD pipeline
- Support for Python 3.9, 3.10, 3.11

## [0.1.1] - 2024-01-08

### Changed
- Improved PyPI metadata with additional keywords and classifiers