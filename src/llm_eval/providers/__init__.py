from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """テキスト生成の設定"""
    max_tokens: int = 512
    temperature: float = 0.0
    system_prompt: str = "You are a helpful assistant."


class BaseProvider(ABC):
    """すべてのLLMプロバイダの基底クラス"""

    def __init__(self, model: str, config: GenerationConfig | None = None):
        self.model = model
        self.config = config or GenerationConfig()

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


from llm_eval.providers.openai_provider import OpenAIProvider  # noqa: E402
from llm_eval.providers.anthropic_provider import AnthropicProvider  # noqa: E402

__all__ = ["BaseProvider", "GenerationConfig", "OpenAIProvider", "AnthropicProvider"]