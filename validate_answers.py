import csv
import os
import sys


def validate_csv_files(directory: str) -> None:
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        print(f"Processing file: {csv_file}")
        try:
            with open(file_path) as f:
                reader = csv.DictReader(f)

                # Check if required columns exist
                required_columns = {"Answer", "Food A", "Food B"}
                if not required_columns.issubset(reader.fieldnames):  # type: ignore[arg-type]
                    print(f"Error: File {csv_file} is missing one or more required columns: {required_columns}")
                    continue

                for row_num, row in enumerate(reader, start=2):  # starting at 2 assuming header is line 1
                    # Use strip() to remove any surrounding whitespace
                    answer = row.get("Answer")
                    food_a = row.get("Food A")
                    food_b = row.get("Food B")

                    if answer not in (food_a, food_b):
                        print(
                            f"Error in file '{csv_file}', row {row_num}: Answer '{answer}' "
                            f"is not equal to Food A '{food_a}' or Food B '{food_b}'.",
                        )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to process file '{csv_file}': {e}")


if __name__ == "__main__":
    # Specify the directory containing CSV files
    directory_path = sys.argv[1]  # Change this to your actual directory path
    validate_csv_files(directory_path)
