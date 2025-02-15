import os

import anthropic
from google import genai
from openai import OpenAI


class GeneralClient:
    """
    Simple class for other clients to inherit from
    """

    def __init__(self) -> None:
        pass

    def call_model(self, message: str, model: str) -> str | None:
        """
        Abstract method to be implemented by subclasses.
        """
        error_code = "Subclasses must implement call_model()"
        raise NotImplementedError(error_code)

    def test(self, model: str) -> None:
        """
        Test whether a given API integration is working for a given model
        """
        reply = self.call_model("How are you today?", model)
        print(model, "\t", reply)


class OpenAIClient(GeneralClient):
    """
    Interface for OpenAI-schema APIs
    """

    def __init__(self, api_key: str = "OPENAI_API_KEY", base_url: str | None = None) -> None:
        self.client = OpenAI(
            api_key=os.environ[api_key],
            base_url=base_url,
        )

    def call_model(self, message: str, model: str) -> str | None:
        """
        Call model, provide LLM response
        """
        response = self.client.chat.completions.create(
            model=model,
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

    def __init__(self) -> None:
        super().__init__(api_key="TOGETHER_AI_API_KEY", base_url="https://api.together.xyz/v1")


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek API is built to be identical with OpenAI API
    """

    def __init__(self) -> None:
        super().__init__(api_key="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com")


class AnthropicClient(GeneralClient):
    """
    Interface for Anthropic LLMs
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    def call_model(self, message: str, model: str) -> str:
        """
        Call model, provide LLM response
        """
        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}],
        )
        return response.content[0].text  # type: ignore[union-attr]


class GoogleClient(GeneralClient):
    """
    Interface for Google LLMs
    """

    from google import genai

    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def call_model(self, message: str, model: str) -> str:
        """
        Call model, provide LLM response
        """
        response = self.client.models.generate_content(model=model, contents=message)
        return response.text  # type: ignore[no-any-return]


OPENAI = OpenAIClient()
DEEPSEEK = DeepSeekClient()
ANTHROPIC = AnthropicClient()
GOOGLE = GoogleClient()
TOGETHER = TogetherAIClient()


def test_clients() -> None:
    """
    Test out the integrations
    """
    # OPENAI.test("gpt-4o")
    # DEEPSEEK.test("deepseek-chat")
    # ANTHROPIC.test("claude-3-5-sonnet-20241022")
    # GOOGLE.test("gemini-2.0-flash")
    # GOOGLE.test("gemini-2.0-flash-thinking-exp-01-21") only 10 RPM
    GOOGLE.test("gemini-2.0-pro-exp-02-05")
    # TOGETHER.test("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")


if __name__ == "__main__":
    test_clients()
