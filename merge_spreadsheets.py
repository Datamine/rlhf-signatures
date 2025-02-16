import sys

import pandas as pd

# File paths for your CSV answersfiles
questions_file = "questions.csv"
answers_file = sys.argv[1]
output_file = f"merged_{answers_file}"

# --- Step 1: Read the questions CSV ---
# Assuming the questions CSV has a header row
questions_df = pd.read_csv(questions_file)

# Clean whitespace from the 'Question' column in questions_df
questions_df["Question"] = questions_df["Question"].str.strip()

# --- Step 2: Read the answers CSV ---
# Since the answers CSV seems to have formatting issues (e.g., extra header-like row),
# we read it without assuming the first row is a header, and then assign column names.
answers_df = pd.read_csv(answers_file, header=None, names=["Question", "Answer"])

# Clean whitespace from the 'Question' column in answers_df
answers_df["Question"] = answers_df["Question"].str.strip()

# --- Step 3: Remove any rows that are likely spurious headers ---
# For example, drop rows where the Question column literally equals "Question"
answers_df = answers_df[answers_df["Question"].str.lower() != "question"]

# Optionally, if you know that some rows in the answers CSV are not valid,
# you can add further filtering here.

# --- Step 4: Merge the answers into the questions DataFrame ---
# Use a left join so that every question is retained.
merged_df = pd.merge(questions_df, answers_df, on="Question", how="left")

# --- Step 5: Write the merged DataFrame to a new CSV file ---
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved as {output_file}")
