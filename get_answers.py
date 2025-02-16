import _csv
import asyncio
import csv
import os
import re
from typing import TextIO

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
    writer: _csv.Writer,
    lock: asyncio.Lock,
    processed_set: set[str],
    csv_file: TextIO,
) -> None:
    """
    Process an individual question
    """
    # Check (with the lock) if the question is already processed.
    async with lock:
        if question in processed_set:
            print(f"[{model_instance.model}] Skipping already processed question: {question}")
            return

    try:
        # If call_model is blocking, run it in a thread.
        answer = await asyncio.to_thread(model_instance.call_model, question)

        # Write the answer immediately (with the lock).
        async with lock:
            # Double-check in case it got processed while waiting.
            if question in processed_set:
                print(
                    f"[{model_instance.model}] Question already processed after API call: {question}",
                )
                return
            writer.writerow([question, answer])
            csv_file.flush()  # write to disk immediately
            processed_set.add(question)
            print(f"[{model_instance.model}] Written: {question} -> {answer}")

    except Exception as e:  # noqa:BLE001
        # If an exception occurs, check if the question was already written.
        async with lock:
            if question in processed_set:
                print(
                    f"[{model_instance.model}] Exception occurred but question already processed: {question}",
                )
            else:
                print(f"[{model_instance.model}] Exception processing question '{question}': {e}")


async def process_model_instance(
    model_instance: GeneralClient,
    question_rows: list[list[str]],
) -> None:
    """
    For a given model instance, process all questions and write answers to answers_[model].csv.
    """
    filename = safe_filename(f"answers_{model_instance.model}.csv")
    processed_set = set()

    # If the file already exists, load questions that have been answered.
    if os.path.exists(filename):
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # row[3] is the actual question itself
                    processed_set.add(row[3])

    # Open the output CSV in append mode.
    csv_file = open(filename, "a", newline="")
    writer = csv.writer(csv_file)

    # Create a lock to control access to the file and the processed_set.
    lock = asyncio.Lock()

    # Create a task for each question (if not already processed).
    tasks = []
    for question_row in question_rows:
        question = question_row[3]
        # question row is a csv row, the question itself is question[3]
        async with lock:
            if question in processed_set:
                continue
        task = asyncio.create_task(
            process_question(model_instance, question, writer, lock, processed_set, csv_file),
        )
        tasks.append(task)

    # Await completion of all tasks for this API instance.
    await asyncio.gather(*tasks)
    csv_file.close()


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
