import os

from openai import OpenAI


class OpenAIClient:
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


OPENAI = OpenAIClient()
DEEPSEEK = DeepSeekClient()


def test_clients() -> None:
    """
    Test out the integrations
    """
    print("GPT-4o", OPENAI.call_model("How are you today?", "gpt-4o"))
    print("Deepseek-chat", DEEPSEEK.call_model("How are you today?", "deepseek-chat"))


if __name__ == "__main__":
    test_clients()
