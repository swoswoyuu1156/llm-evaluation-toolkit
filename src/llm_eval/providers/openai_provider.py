import os

from llm_eval.providers import BaseProvider, GenerationConfig


class OpenAIProvider(BaseProvider):
    """OpenAI API プロバイダ"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: GenerationConfig | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model=model, config=config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openaiパッケージが必要です: pip install llm-evaluation-toolkit[openai]"
            )
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""