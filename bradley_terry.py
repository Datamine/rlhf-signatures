import glob
import os
import sys

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.typing import NDArray


def estimate_bt(
    wins: NDArray[np.float64],
    contests: NDArray[np.float64],
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> NDArray[np.float64]:
    """
    Iteratively solve for the Bradley-Terry abilities (pi) given:
      wins[i] = total wins for option i,
      contests[i, j] = total number of contests between options i and j.

    The fixed-point update is:
      pi[i] = wins[i] / sum_{j != i} { contests[i,j] / (pi[i] + pi[j]) }

    Normalization is applied (here we rescale so that sum(pi)=1) for stability.
    """
    n = len(wins)
    # Create an array of ones with dtype float64 for consistency
    pi: NDArray[np.float64] = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        pi_old = pi.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and contests[i, j] > 0:
                    denom += contests[i, j] / (pi_old[i] + pi_old[j])
            # Avoid division by zero:
            if denom > 0:
                pi[i] = wins[i] / denom
            else:
                pi[i] = 0.0
        # Normalize (this does not change the relative ranking)
        pi = pi / np.sum(pi)
        if np.max(np.abs(pi - pi_old)) < tol:
            break
    return pi


def compute_hessian(beta: NDArray[np.float64], contests: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the Hessian (matrix of second derivatives) for the log-likelihood,
    evaluated at the estimated beta = log(pi). For a pair (i,j) with total contests n_ij,
    the contribution is:

      H_{ii} = -sum_{j != i} n_ij * (exp(beta_i)*exp(beta_j)) / (exp(beta_i)+exp(beta_j))^2
      H_{ij} =  n_ij * (exp(beta_i)*exp(beta_j)) / (exp(beta_i)+exp(beta_j))^2, for i != j.
    """
    n = len(beta)
    exp_beta = np.exp(beta)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and contests[i, j] > 0:
                term = contests[i, j] * (exp_beta[i] * exp_beta[j]) / ((exp_beta[i] + exp_beta[j]) ** 2)
                hessian[i, j] = term
        # Diagonal: sum over all opponents j (i ≠ j)
        row_sum = 0.0
        for j in range(n):
            if i != j:
                row_sum += contests[i, j] * (exp_beta[i] * exp_beta[j]) / ((exp_beta[i] + exp_beta[j]) ** 2)
        hessian[i, i] = -row_sum
    return hessian


def compute_bt_for_file(filename: str) -> list[dict]:  # type: ignore[type-arg]
    # Read the CSV file.
    df = pd.read_csv(filename)
    required_columns = ["Option 1", "Option 2", "Answer"]
    if not all(col in df.columns for col in required_columns):
        print("CSV file must contain columns:", required_columns)
        sys.exit(1)

    # Get the list of unique options (from both Option 1 and Option 2)
    options = pd.concat([df["Option 1"], df["Option 2"]]).unique()
    options = sorted(options)  # sort for consistency
    n = len(options)
    option_to_index = {option: i for i, option in enumerate(options)}

    # Initialize arrays to hold win counts and contest counts.
    wins = np.zeros(n)
    contests = np.zeros((n, n))

    # Process each row of the dataframe.
    # Each row is assumed to represent a contest between Option 1 and Option 2.
    # 'Answer' should be equal to one of these.
    for _, row in df.iterrows():
        opt1 = row["Option 1"]
        opt2 = row["Option 2"]
        winner = row["Answer"]
        i = option_to_index[opt1]
        j = option_to_index[opt2]
        # Increment contest counts (both ways since the contest is between i and j)
        contests[i, j] += 1
        contests[j, i] += 1
        # Increment win count for the winning option.
        if winner == opt1:
            wins[i] += 1
        elif winner == opt2:
            wins[j] += 1
        else:
            print(f"Warning: Invalid Answer in row:\n{row}")

    # Estimate the ability parameters using the iterative algorithm.
    pi_est = estimate_bt(wins, contests)

    # Transform to log-scale: beta = log(pi).
    # For identifiability, fix the first option as reference (i.e. beta[0] = 0).
    beta_est = np.log(pi_est)
    beta_est = beta_est - beta_est[0]

    # Compute the Hessian matrix at the estimated beta.
    hessian = compute_hessian(beta_est, contests)

    # Since the model is invariant to an overall shift, we fix beta[0]=0.
    # Remove the first row and column to invert the (n-1)x(n-1) observed information.
    hessian_free = hessian[1:, 1:]
    try:
        # The observed Fisher information is -H; its inverse approximates the covariance.
        cov_free = inv(-hessian_free)
    except np.linalg.LinAlgError:
        print(f"Error for {filename}: Hessian is singular; cannot compute confidence intervals.")
        cov_free = np.full(hessian_free.shape, np.nan)

    # Build the standard errors vector.
    se = np.zeros(n)
    se[0] = 0  # the reference is fixed (no variance)
    for i in range(1, n):
        se[i] = np.sqrt(cov_free[i - 1, i - 1])

    # Compute approximate 95% confidence intervals for beta, and then for the ability (pi).
    z = 1.96
    beta_lower = beta_est - z * se
    beta_upper = beta_est + z * se
    # Transform back to the ability scale.
    pi_lower = np.exp(beta_lower)
    pi_upper = np.exp(beta_upper)

    # Prepare and display the results.
    results = []
    for i, option in enumerate(options):
        results.append(
            {
                "Option": option,
                "Ability (pi)": np.exp(beta_est[i]),  # ability relative to the reference
                "Beta": beta_est[i],
                "SE (Beta)": se[i],
                "Beta 95% CI": (beta_lower[i], beta_upper[i]),
                "Ability 95% CI": (pi_lower[i], pi_upper[i]),
            },
        )

    # Sort by estimated ability (highest first)
    return sorted(results, key=lambda x: x["Ability (pi)"], reverse=True)


if __name__ == "__main__":
    user_arg = sys.argv[1]
    if user_arg.endswith(".csv"):
        filename = sys.argv[1]
        results = compute_bt_for_file(filename)
        print("\nBradley-Terry Rankings:")
        for res in results:
            print(
                f"{res['Option']}: Ability = {res['Ability (pi)']:.3f}, "
                f"95% CI = ({res['Ability 95% CI'][0]:.3f}, {res['Ability 95% CI'][1]:.3f}), "
                f"Beta = {res['Beta']:.3f} (SE = {res['SE (Beta)']:.3f})",
            )
    else:
        # Otherwise, assume the argument is a directory.
        csv_files = glob.glob(os.path.join(user_arg, "*.csv"))
        if not csv_files:
            print("No CSV files found in the specified directory.")
            sys.exit(1)

        # Dictionary to hold results: { model_name: { Option -> Ability } }
        model_results = {}
        for file_path in csv_files:
            # unclean code, but skip the 2.0-pro-exp file because it's bad data (due to ratelimit)z
            if "2.0-pro-exp" in file_path:
                continue

            # Model name is taken from the filename (without extension)
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                bt_results = compute_bt_for_file(file_path)
                # Extract a mapping from Option to its Ability (pi)
                option_to_ability = {entry["Option"]: entry["Ability (pi)"] for entry in bt_results}
                model_results[model_name] = option_to_ability
            except Exception as e:  # noqa: BLE001
                print(f"Error processing file {file_path}: {e}", file=sys.stderr)

        # Create a DataFrame where rows are choice options and columns are model names.
        df_results = pd.DataFrame(model_results)
        df_results.sort_index(inplace=True)  # noqa: PD002

        print("Bradley-Terry Ability Scores:")
        print(df_results.to_string(float_format=lambda x: f"{x:.3f}"))

        # Save the results to a CSV file.
        output_csv = "bradley_terry.csv"
        df_results.to_csv(output_csv, float_format="%.3f")
        print(f"Results saved to {output_csv}")
