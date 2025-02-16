import asyncio
import csv
import re

from llm_interface import ALL_MODELS, GeneralClient


def safe_filename(filename: str) -> str:
    """
    Sanitize the filename by replacing unsafe characters with underscores.
    Only allows letters, digits, underscores, hyphens, and dots.
    """
    return re.sub(r"[^\w\-.]", "_", filename)


async def process_question(
    model_instance: GeneralClient,
    question: str,
    csv_filename: str,
) -> bool:
    """
    Process an individual question:
    - Open the CSV file and check if the question (in column 0) is already present.
    - If not, call the model and append the [question, answer] row.
    - return true/false depending on whether api was called
    """
    # Check on disk if the question is already processed.
    try:
        with open(csv_filename, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                # question is stored in row[0]
                if row and row[0] == question:
                    print(f"[{model_instance.model}] Skipping: {question}")
                    return False
    except FileNotFoundError:
        # The file does not exist yet, so no processed questions.
        pass

    try:
        # Run the (potentially blocking) model call in a separate thread.
        answer = await asyncio.to_thread(model_instance.call_model, question)

        # Append the answer to the CSV file.
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([question, answer])
        print(f"[{model_instance.model}] Written: {question} -> {answer}")

    except Exception as e:  # noqa: BLE001
        print(f"[{model_instance.model}] Exception: '{question}': {e}")
    return True


async def process_model_instance(
    model_instance: GeneralClient,
    question_rows: list[list[str]],
) -> None:
    """
    For a given model instance, process all questions and write answers to answers_[model].csv.
    """
    csv_filename = safe_filename(f"answers_{model_instance.model}.csv")

    if model_instance.rate_limit_between_calls:
        for question_row in question_rows:
            # The question is assumed to be in the fourth column (index 3) of the questions CSV.
            question = question_row[3]
            api_called = await process_question(model_instance, question, csv_filename)
            if api_called:
                # avoid whatever rate limits apply
                await asyncio.sleep(model_instance.rate_limit_between_calls)
    else:
        # Process questions concurrently.
        tasks = []
        for question_row in question_rows:
            question = question_row[3]
            tasks.append(asyncio.create_task(process_question(model_instance, question, csv_filename)))
        await asyncio.gather(*tasks)


def load_questions() -> list[list[str]]:
    """
    Synchronously load questions from 'questions.csv' and append an empty "Answer" field.
    """
    questions = []
    with open("questions.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header row
        header.append("Answer")  # Add the new "Answer" column
        questions.append(header)

        for row in reader:
            row.append("")  # Append an empty string as the answer field
            questions.append(row)
    return questions


async def process_all_questions(questions: list[list[str]]) -> None:
    """
    Asynchronously process the questions using all API model instances.
    """
    tasks = []
    for model_instance in ALL_MODELS:
        tasks.append(asyncio.create_task(process_model_instance(model_instance, questions)))  # noqa: PERF401
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # First, synchronously load the questions from CSV
    questions = load_questions()

    # Then, use asyncio to process the questions concurrently
    asyncio.run(process_all_questions(questions))
