import sys

import pandas as pd
from scipy.stats import ttest_rel


def compute_paired_order_bias(filename: str) -> None:
    """
    Performs a paired t-test to evaluate whether the mean difference in proportions
    (between two orderings) is statistically different from zero. A mean difference
    of zero would indicate that the ordering has no effect.

    Returns:
        t_statistic (float): The t-statistic, representing how many standard errors
                            the observed mean difference is away from zero.
        p_value (float): The p-value for the test. A p-value less than 0.05 suggests
                        a statistically significant order effect.

    Example:
        A t-statistic of -3 and a p-value of 0.04 indicates that the mean difference is
        about three standard errors below zero, providing statistically significant
        evidence (p < 0.05) of an order effect.
    """

    try:
        df = pd.read_csv(filename)
    except Exception as e:  # noqa: BLE001
        print("Error reading file:", e)
        return

    # Ensure the required columns exist
    required_columns = {"Option 1", "Option 2", "Answer"}
    if not required_columns.issubset(df.columns):
        print(f"Error: The CSV file must contain columns: {required_columns}")
        return

    # Create a canonical identifier for each food pair (order-independent)
    # Here, we simply sort the names to create a tuple key.
    df["pair_id"] = df.apply(lambda row: tuple(sorted([str(row["Option 1"]), str(row["Option 2"])])), axis=1)

    # Assign an order indicator:
    # order 0: Option 1 equals the first element in the sorted (canonical) pair.
    # order 1: Otherwise.
    df["order"] = df.apply(
        lambda row: 0 if str(row["Option 1"]) == sorted([str(row["Option 1"]), str(row["Option 2"])])[0] else 1,
        axis=1,
    )

    # Create a binary indicator: True if Answer equals Option 1 (i.e., the first column)
    df["first_selected"] = df["Answer"] == df["Option 1"]

    # Group the data by the canonical food pair.
    grouped = df.groupby("pair_id")

    # For each pair, compute the proportion of responses in which the first column was selected,
    # separately for order 0 and order 1.
    results = []
    for pair, group in grouped:
        orders = group["order"].unique()
        if 0 not in orders or 1 not in orders:
            print(f"Skipping pair {pair} because it does not have both order 0 and order 1 responses.")
            continue

        group_order0 = group[group["order"] == 0]
        group_order1 = group[group["order"] == 1]
        p0 = group_order0["first_selected"].mean()  # Proportion for order 0
        p1 = group_order1["first_selected"].mean()  # Proportion for order 1

        results.append(
            {
                "pair_id": pair,
                "p0": p0,
                "p1": p1,
                "n0": len(group_order0),
                "n1": len(group_order1),
            },
        )

    if not results:
        print("No pairs with responses in both orders found. Exiting.")
        return

    res_df = pd.DataFrame(results)
    print("Summary of paired proportions by food pair (order 0 vs. order 1):")
    print(res_df[["pair_id", "p0", "p1", "n0", "n1"]])

    # Perform a paired t-test on the proportions across food pairs.
    t_stat, p_val = ttest_rel(res_df["p0"], res_df["p1"])

    print("\nPaired t-test on the proportions (order 0 minus order 1):")
    print("  t-statistic:", t_stat)
    print(f"  p-value:    {p_val:.3f}")


if __name__ == "__main__":
    filename = sys.argv[1]
    compute_paired_order_bias(filename)
