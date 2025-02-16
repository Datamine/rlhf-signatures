import os
from typing import Optional

import anthropic
from google import genai
from openai import OpenAI


class GeneralClient:
    """
    Simple class for other clients to inherit from
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def call_model(self, message: str, model: Optional[str] = None) -> str | None:
        """
        Abstract method to be implemented by subclasses.
        """
        error_code = "Subclasses must implement call_model()"
        raise NotImplementedError(error_code)

    def test(self, model: Optional[str] = None) -> None:
        """
        Test whether a given API integration is working for a given model
        """
        reply = self.call_model(
            "What is the more delicious food, Jollof Rice or Pepperoni Pizza? "
            "Answer with only and exactly one of these two options.",
            model,
        )
        print(model or self.model, "\t", reply)


class OpenAIClient(GeneralClient):
    """
    Interface for OpenAI-schema APIs
    """

    def __init__(
        self,
        model: str,
        api_key: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=os.environ[api_key],
            base_url=base_url,
        )
        self.model = model

    def call_model(self, message: str, override_model: Optional[str] = None) -> str | None:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": message},
            ],
            stream=False,
        )
        # this never provides more than 1 choice, unless we change the request parameters
        return response.choices[0].message.content


class TogetherAIClient(OpenAIClient):
    """
    together.ai API is built to be identical with OpenAI API
    """

    def __init__(self, model: str) -> None:
        super().__init__(
            model,
            api_key="TOGETHER_AI_API_KEY",
            base_url="https://api.together.xyz/v1",
        )


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek API is built to be identical with OpenAI API
    """

    def __init__(self, model: str) -> None:
        super().__init__(model, api_key="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")


class AnthropicClient(GeneralClient):
    """
    Interface for Anthropic LLMs
    """

    def __init__(self, model: str) -> None:
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def call_model(self, message: str, override_model: Optional[str] = None) -> str:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model
        response = self.client.messages.create(
            model=model_to_use,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text  # type: ignore[union-attr]


class GoogleClient(GeneralClient):
    """
    Interface for Google LLMs
    """

    def __init__(self, model: str) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def call_model(self, message: str, override_model: Optional[str] = None) -> str:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model
        response = self.client.models.generate_content(model=model_to_use, contents=message)
        return response.text  # type: ignore[no-any-return]


OPENAI_GPT_4O = OpenAIClient("gpt-4o")
OPENAI_O1 = OpenAIClient("o1")
DEEPSEEK_V3 = DeepSeekClient("deepseek-chat")
DEEPSEEK_R1 = DeepSeekClient("deepseek-reasoner")
CLAUDE_35 = AnthropicClient("claude-3-5-sonnet-20241022")
GEMINI_2_0_FLASH = GoogleClient("gemini-2.0-flash")
GEMINI_2_0_PRO = GoogleClient("gemini-2.0-pro-exp-02-05")
LLAMA_405B = TogetherAIClient("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")

ALL_MODELS = [
    OPENAI_GPT_4O,
    OPENAI_O1,
    DEEPSEEK_R1,
    DEEPSEEK_V3,
    CLAUDE_35,
    GEMINI_2_0_FLASH,
    GEMINI_2_0_PRO,
    LLAMA_405B,
]


def test_clients() -> None:
    """
    Test out the integrations
    """
    for a in ALL_MODELS:
        a.test()


if __name__ == "__main__":
    test_clients()
