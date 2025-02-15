import os

import anthropic
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

        # TODO: when does this provide several choices?
        return response.choices[0].message.content


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


OPENAI = OpenAIClient()
DEEPSEEK = DeepSeekClient()
ANTHROPIC = AnthropicClient()


def test_clients() -> None:
    """
    Test out the integrations
    """
    OPENAI.test("gpt-4o")
    DEEPSEEK.test("deepseek-chat")
    ANTHROPIC.test("claude-3-5-sonnet-20241022")


if __name__ == "__main__":
    test_clients()
