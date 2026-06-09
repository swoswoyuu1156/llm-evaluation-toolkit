import os

from llm_eval.providers import BaseProvider, GenerationConfig


class AnthropicProvider(BaseProvider):
    """Anthropic API プロバイダ"""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        config: GenerationConfig | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model=model, config=config)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropicパッケージが必要です: pip install llm-evaluation-toolkit[anthropic]"
            )
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.content[0].text