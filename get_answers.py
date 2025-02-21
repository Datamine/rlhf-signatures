import argparse
import asyncio
import sys

from llm_interface import ALL_MODELS, GeneralClient
from redis_interface import SpreadsheetRedisProcessor


def process_question(
    model: GeneralClient,
    question: str,
) -> str:
    try:
        return model.call_model(question)
    except Exception as e:  # noqa: BLE001
        print(f"[{model.model_name}] Exception: '{question}': {e}")
        return ""


async def process_model_instance(
    processor: SpreadsheetRedisProcessor,
    model: GeneralClient,
) -> None:
    """
    For a given model instance, process all questions and write answers to answers_[model].csv.
    The output file will include the original question row (all columns) with the answer in the final column.
    """
    question = processor.get_next_unprocessed_question(model)
    while question:
        # synchronously process questions because of LLM api rate limits
        # could be made async if desired
        answer = process_question(model, question)
        processor.set_answer(model_instance, question, answer)
        await asyncio.sleep(model_instance.rate_limit_between_calls)


async def process_all_questions(processor: SpreadsheetRedisProcessor) -> None:
    """
    Asynchronously process the questions using all API model instances.
    """
    tasks = []
    for model_instance in processor.llm_models:
        tasks.append(  # noqa:PERF401
            asyncio.create_task(process_model_instance(processor, model_instance)),
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interface for working with questions for LLMs.",
    )
    parser.add_argument("--questions_filename", type=str, help="Filename of the questions spreadsheet", required=False)
    parser.add_argument("--model", type=str, help="Run for only this model", required=False)
    parser.add_argument("--stats", action="store_true", help="Display statistics")
    parser.add_argument("--new", action="store_true", help="Clear all prior question values")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save all current values to a spreadsheet (needs --questions_filename)",
    )

    args = parser.parse_args()

    if args.model:
        for model_instance in ALL_MODELS:
            if args.model == model_instance.model_name:
                models_to_use = [model_instance]
                break
        else:
            raise ValueError("Improper model name provided.")  # noqa:TRY003,EM101
    else:
        models_to_use = ALL_MODELS

    processor = SpreadsheetRedisProcessor(models_to_use)

    if args.stats:
        processor.stats()
        sys.exit(0)

    processor.clear_all_locks()
    processor.clear_processing()

    if args.questions_filename:
        if args.save:
            processor.export_answers(args.questions_filename)

        if args.new:
            processor.clear_all()

        # by default, we don't delete any existing answers in case we're continuing a prior workflow
        processor.load_questions(args.questions_filename)

    # Then, use asyncio to process the questions concurrently.
    asyncio.run(process_all_questions(processor))
