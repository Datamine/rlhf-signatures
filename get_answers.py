import asyncio
import csv
import os
import re
import sys
from typing import Optional

from llm_interface import ALL_MODELS, GeneralClient


def safe_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing unsafe characters with underscores.
    Only allows letters, digits, underscores, hyphens, and dots.
    """
    return re.sub(r"[^\w\-.]", "_", filename)


async def process_question(
    model_instance: GeneralClient,
    question_row: list[str],
    csv_filename: str,
) -> bool:
    """
    Process an individual question row:
    - Check if the question (assumed to be in column index 3) is already present (ignoring header).
    - If not, call the model and append the entire row with the answer filled in the last column.
    - Return True/False depending on whether the API was called.
    """
    question_text = question_row[3]

    # Check on disk if the question is already processed (skip header row).
    try:
        with open(csv_filename, newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header row
            for row in reader:
                # Assuming the question text is stored in column 3
                if len(row) > 3 and row[3] == question_text:  # noqa: PLR2004
                    print(f"[{model_instance.model}] Skipping: {question_text}")
                    return False
    except FileNotFoundError:
        # The file does not exist yet, so no processed questions.
        pass

    try:
        # Run the (potentially blocking) model call in a separate thread.
        answer = await asyncio.to_thread(model_instance.call_model, question_text)

        # Create a copy of the question row and update the last column with the answer.
        row_to_write = question_row.copy()
        row_to_write[-1] = answer  # type: ignore[assignment]

        # Append the full row (with answer) to the CSV file.
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_to_write)
        print(f"[{model_instance.model}] Written: {question_text} -> {answer}")

    except Exception as e:  # noqa: BLE001
        print(f"[{model_instance.model}] Exception: '{question_text}': {e}")
    return True


async def process_model_instance(
    model_instance: GeneralClient,
    question_rows: list[list[str]],
) -> None:
    """
    For a given model instance, process all questions and write answers to answers_[model].csv.
    The output file will include the original question row (all columns) with the answer in the final column.
    """
    csv_filename = safe_filename(f"{model_instance.model}.csv")

    # Write header row if the file doesn't exist.
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(question_rows[0])

    # Skip header row when processing questions.
    data_rows = question_rows[1:]

    if model_instance.rate_limit_between_calls:
        for question_row in data_rows:
            api_called = await process_question(model_instance, question_row, csv_filename)
            if api_called:
                # Respect the rate limit.
                await asyncio.sleep(model_instance.rate_limit_between_calls)
    else:
        # Process questions concurrently.
        tasks = [
            asyncio.create_task(process_question(model_instance, question_row, csv_filename))
            for question_row in data_rows
        ]
        await asyncio.gather(*tasks)


def load_questions(questions_filename: str) -> list[list[str]]:
    """
    Synchronously load questions from 'questions.csv' and append an empty "Answer" field.
    The first row is assumed to be the header.
    """
    questions = []
    with open(questions_filename, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header row
        header.append("Answer")  # Add the new "Answer" column
        questions.append(header)

        for row in reader:
            row.append("")  # Append an empty string for the answer field
            questions.append(row)
    return questions


async def process_all_questions(questions: list[list[str]], optional_model: Optional[str] = None) -> None:
    """
    Asynchronously process the questions using all API model instances.
    """
    tasks = []
    for model_instance in ALL_MODELS:
        # if a specific model has been selected, only pick those tasks
        if optional_model and optional_model != model_instance.model:
            continue
        tasks.append(
            asyncio.create_task(process_model_instance(model_instance, questions)),
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # optional model selection
    questions_filename = sys.argv[1]
    optional_model = sys.argv[2] if len(sys.argv) > 2 else None

    # First, synchronously load the questions from CSV.
    questions = load_questions(questions_filename)

    # Then, use asyncio to process the questions concurrently.
    asyncio.run(process_all_questions(questions, optional_model=optional_model))
