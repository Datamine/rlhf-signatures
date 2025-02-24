import argparse

from llm_interface import ALL_MODELS


def main() -> str:
    """
    Re-ask a question to a given model, in case the answers provided in a spreadsheet are not valid.
    This bypasses any Redis cache and handles synchronously.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process a sentence with an optional model parameter.")

    # Positional argument for the sentence
    parser.add_argument("question", type=str, help="The question input in quotes.")
    parser.add_argument("--model", type=str, help="Specify the model to use (default: default_model).")

    args = parser.parse_args()
    question = args.question
    model = [x for x in ALL_MODELS if x.model_name == args.model][0]  # noqa: RUF015
    return model.call_model(question)


if __name__ == "__main__":
    print(main())
