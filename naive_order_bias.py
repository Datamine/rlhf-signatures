import sys

import pandas as pd
from scipy.stats import binomtest, chisquare


def compute_naive_order_bias(filename: str) -> None:
    """
    This will run a binomial test and a chi-square test on the null hypothesis that
    there is no bias with respect to order of the first/second result

    p-values below 0.05 indicate bias in how the answers are recorded relative to the order of options
    """
    try:
        df = pd.read_csv(filename)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading the file: {e}")
        return

    # Check that required columns exist
    required_columns = {"Food A", "Food B", "Answer"}
    if not required_columns.issubset(df.columns):
        print(f"Error: The CSV file must contain the columns: {required_columns}")
        return

    # Count how many times the answer equals Food A or Food B
    first_count = (df["Answer"] == df["Food A"]).sum()
    second_count = (df["Answer"] == df["Food B"]).sum()

    total = first_count + second_count

    print("Total valid comparisons (where Answer matched Food A or Food B):", total)
    print("Number of times Answer equals 'Food A':", first_count)
    print("Number of times Answer equals 'Food B':", second_count)

    # Binomial test:
    # Under the null hypothesis, the probability that the answer is in position A is 0.5.
    p_val_binom = binomtest(first_count, n=total, p=0.5, alternative="two-sided")
    print("\nBinomial Test:")
    print(f"  Proportion of 'Food A' answers: {first_count / total:.3f}")
    print(f"  p-value: {p_val_binom.pvalue:.3f}")

    # Chi-square goodness-of-fit test:
    # Expected counts are [total/2, total/2] if there's no bias.
    observed = [first_count, second_count]
    expected = [total / 2, total / 2]
    chi2_stat, p_val_chi2 = chisquare(f_obs=observed, f_exp=expected)
    print("\nChi-square Goodness-of-Fit Test:")
    print(f"  Chi-square statistic: {chi2_stat:.3f}")
    print(f"  p-value: {p_val_chi2:.3f}")


if __name__ == "__main__":
    filename = sys.argv[1]
    compute_naive_order_bias(filename)
