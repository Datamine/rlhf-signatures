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
                required_columns = {"Answer", "Option 1", "Option 2"}
                if not required_columns.issubset(reader.fieldnames):  # type: ignore[arg-type]
                    print(f"Error: File {csv_file} is missing one or more required columns: {required_columns}")
                    continue

                for row_num, row in enumerate(reader, start=2):  # starting at 2 assuming header is line 1
                    # Use strip() to remove any surrounding whitespace
                    answer = row.get("Answer").strip().strip(".")
                    option_a = row.get("Option 1")
                    option_b = row.get("Option 2")

                    if answer not in (option_a, option_b):
                        print(
                            f"Error in file '{csv_file}', row {row_num}: Answer '{answer}' "
                            f"is not equal to Option 1 '{option_a}' or Option 2 '{option_b}'.",
                        )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to process file '{csv_file}': {e}")
    print("Finished!")

if __name__ == "__main__":
    # Specify the directory containing CSV files
    directory_path = sys.argv[1]  # Change this to your actual directory path
    validate_csv_files(directory_path)
