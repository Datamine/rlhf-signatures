import sys

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def compute_paired_order_bias(filename: str) -> None:
    """
    Run McNemar's test on ordered question pairs (Food A, Food B) vs. (Food B, Food A)
    p value < 0.05 would suggest that ordering influences the choice
    """
    # Load the CSV file
    try:
        df = pd.read_csv(filename)
    except Exception as e:  # noqa: BLE001
        print("Error reading file:", e)
        return

    # Check that required columns exist
    required_columns = {"Food A", "Food B", "Answer"}
    if not required_columns.issubset(df.columns):
        print("Error: The CSV file must contain the columns:", required_columns)
        return

    # Create a canonical identifier for each food pair (order-independent)
    def canonical_pair(row: dict[str, str]) -> tuple[str, ...]:
        # Convert to strings to ensure consistent comparison
        pair = sorted([str(row["Food A"]), str(row["Food B"])])
        return tuple(pair)  # tuple is hashable

    df["pair_id"] = df.apply(canonical_pair, axis=1)

    # Assign an order indicator:
    # order 0: Food A equals the first element in the canonical (sorted) pair.
    # order 1: Otherwise.
    def assign_order(row: dict[str, str]) -> int | None:
        canon = list(row["pair_id"])
        if str(row["Food A"]) == canon[0]:
            return 0
        if str(row["Food A"]) == canon[1]:
            return 1

        return None

    df["order"] = df.apply(assign_order, axis=1)

    # Create a binary indicator: True if Answer equals Food A (i.e. first column)
    df["first_selected"] = df["Answer"] == df["Food A"]

    # Group by the canonical food pair and only consider groups with exactly 2 responses.
    pairs = df.groupby("pair_id")

    # Initialize counts for the contingency table:
    # a: both orders selected the first column.
    # b: canonical (order 0) selected first; reversed (order 1) did not.
    # c: canonical did not select first; reversed did.
    # d: neither selected first.
    a = b = c = d = 0
    pair_count = 0

    for pair, group in pairs:
        if len(group) != 2:  # noqa: PLR2004
            print(f"Skipping pair {pair} because it does not have 2 entries (has {len(group)}).")
            continue

        # Ensure one row is order 0 and one is order 1.
        row_order0 = group[group["order"] == 0]
        row_order1 = group[group["order"] == 1]
        if row_order0.empty or row_order1.empty:
            print(f"Skipping pair {pair} because it doesn't have both order 0 and order 1.")
            continue

        # Get the binary outcome for each ordering
        val0 = row_order0.iloc[0]["first_selected"]
        val1 = row_order1.iloc[0]["first_selected"]

        if val0 and val1:
            a += 1
        elif val0 and not val1:
            b += 1
        elif not val0 and val1:
            c += 1
        elif not val0 and not val1:
            d += 1
        pair_count += 1

    print("Total paired food combinations analyzed:", pair_count)
    print("Contingency table counts (across pairs):")
    print("  Both orders: first column chosen (a):", a)
    print("  Canonical only (b):", b)
    print("  Reversed only (c):", c)
    print("  Neither (d):", d)

    # McNemar's test focuses on the discordant pairs (b and c)
    if (b + c) == 0:
        print("\nNo discordant pairs found. McNemar's test is not applicable.")
        p_value = 1.0
    else:
        # Use the exact binomial test (appropriate for small sample sizes)
        result = mcnemar([[a, b], [c, d]], exact=True)
        p_value = result.pvalue
        print("\nMcNemar's test results:")
        print("  Test statistic:", result.statistic)
        print("  p-value:", p_value)


if __name__ == "__main__":
    filename = sys.argv[1]
    compute_paired_order_bias(filename)
