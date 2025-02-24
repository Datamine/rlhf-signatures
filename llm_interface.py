import asyncio
import os
import time
from typing import Any, Optional

import anthropic
from google import genai
from openai import OpenAI


class GeneralClient:
    """
    Simple class for other clients to inherit from
    """

    def __init__(
        self,
        model: str,
        rate_limit_between_calls: int = 0,
        api_key: str = "",
        base_url: Optional[str] = None,
        measure_performance: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.model_name = model
        self.rate_limit_between_calls = rate_limit_between_calls
        self.api_key = api_key
        self.base_url = base_url
        self.measure_performance = measure_performance

    def _call_model(self, message: str, override_model: Optional[str] = None) -> str:
        raise NotImplementedError("Must be implemented by child class")  # noqa:EM101

    def call_model(self, message: str, override_model: Optional[str] = None) -> str:
        start_time = time.perf_counter()
        # Call function implemented by child class
        result = self._call_model(message, override_model=override_model)
        latency = time.perf_counter() - start_time
        if self.measure_performance:
            print(f"[{self.model_name}] API call took {latency:.2f} seconds")
        return result

    async def call_model_async(self, message: str, model: Optional[str] = None) -> str:
        """
        Async model call - by default runs sync version in thread pool,
        but subclasses should implement native async if available.
        Logs the latency of the API call.
        """
        return await asyncio.to_thread(self.call_model, message, model)

    def test(self, model: Optional[str] = None) -> None:
        """
        Test whether a given API integration is working for a given model
        """
        reply = self.call_model(
            "What is the more delicious food, Jollof Rice or Pepperoni Pizza? "
            "Answer with only and exactly one of these two options.",
            model,
        )
        print(model or self.model_name, "\t", self.base_url, reply)


class OpenAIClient(GeneralClient):
    """
    Interface for OpenAI-schema APIs
    """

    # TODO: properly type this
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        api_key = kwargs.pop("api_key", "OPENAI_API_KEY")
        super().__init__(*args, **{**kwargs, "api_key": api_key})
        self.client = OpenAI(
            api_key=os.environ[self.api_key],
            base_url=self.base_url,
        )

    def _call_model(self, message: str, override_model: Optional[str] = None) -> str:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model_name
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": message},
            ],
            stream=False,
        )
        # this never provides more than 1 choice, unless we change the request parameters
        return response.choices[0].message.content or ""


class TogetherAIClient(OpenAIClient):
    """
    together.ai API is built to be identical with OpenAI API
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(
            *args,
            **kwargs,
            api_key="TOGETHER_AI_API_KEY",
            base_url="https://api.together.xyz/v1",
        )


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek API is built to be identical with OpenAI API
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(
            *args,
            **kwargs,
            api_key="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
        )


class AnthropicClient(GeneralClient):
    """
    Interface for Anthropic LLMs
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        api_key = kwargs.pop("api_key", "ANTHROPIC_API_KEY")
        super().__init__(*args, **{**kwargs, "api_key": api_key})
        self.client = anthropic.Anthropic(api_key=os.environ[self.api_key])

    def _call_model(self, message: str, override_model: Optional[str] = None) -> str:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model_name
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        api_key = kwargs.pop("api_key", "GEMINI_API_KEY")
        super().__init__(*args, **{**kwargs, "api_key": api_key})
        self.client = genai.Client(api_key=os.environ[self.api_key])

    def _call_model(self, message: str, override_model: Optional[str] = None) -> str:
        """
        Call model, provide LLM response
        """
        # use the override if supplied, otherwise use default from instantiation
        model_to_use = override_model or self.model_name
        response = self.client.models.generate_content(model=model_to_use, contents=message)
        return response.text  # type: ignore[no-any-return]


OPENAI_GPT_4O = OpenAIClient("gpt-4o")
OPENAI_O1 = OpenAIClient("o1")
# Deepseek API is down, replacing with TogetherAI hosted version
DEEPSEEK_V3 = DeepSeekClient("deepseek-chat")
# DEEPSEEK_V3 = TogetherAIClient("deepseek-ai/DeepSeek-V3")
# DEEPSEEK_R1 = DeepSeekClient("deepseek-reasoner")
DEEPSEEK_R1 = TogetherAIClient("deepseek-ai/DeepSeek-R1", rate_limit_between_calls=21)
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
    # GEMINI_2_0_PRO, # don't have the daily RPM
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
