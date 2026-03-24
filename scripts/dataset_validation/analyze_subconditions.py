"""
Analyze subconditions in the AnaphoraGym Subconditions dataset.

This script:
- Loads ONLY the thesis subconditions CSV
- Aggregates counts per (condition, sub_cond):
  - n_items (rows)
  - total_inputs (sum of n_input)
  - total_continuations (sum of n_continuations)
  - total_tests (sum of n_tests)
- Prints a summary table
- Produces a bar plot showing counts by subcondition
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")


def load_thesis_subconditions_csv(project_root: Path) -> pd.DataFrame:
    """Load the thesis subconditions CSV."""
    csv_path = (
        project_root
        / "dataset"
        / "AnaphoraGym_Subconditions.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at: {csv_path}")
    return pd.read_csv(csv_path)


def aggregate_by_subcondition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate counts per (condition, sub_cond).

    Assumes the following columns exist:
    - condition
    - sub_cond
    - n_input
    - n_continuations
    - n_tests
    """
    required_cols = [
        "condition",
        "sub_cond",
        "n_input",
        "n_continuations",
        "n_tests",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    grouped = (
        df.groupby(["condition", "sub_cond"], as_index=False)
        .agg(
            n_items=("item", "nunique"),
            total_inputs=("n_input", "sum"),
            total_continuations=("n_continuations", "sum"),
            total_tests=("n_tests", "sum"),
        )
        .sort_values(["condition", "sub_cond"])
    )

    return grouped


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Pretty-print the summary table to the console."""
    print("\n=== Thesis Subconditions Summary ===\n")
    print(
        summary_df.to_string(
            index=False,
            justify="left",
            col_space=12,
            formatters={
                "n_items": "{:d}".format,
                "total_inputs": "{:d}".format,
                "total_continuations": "{:d}".format,
                "total_tests": "{:d}".format,
            },
        )
    )
    print("\nTotal unique (condition, sub_cond) combinations:", len(summary_df))


def plot_subcondition_counts(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a bar plot showing counts by subcondition.

    X-axis: sub_cond (optionally grouped/colored by condition)
    Y-axis: number of items and tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "thesis_subconditions_counts.png"

    # Build a long-form DataFrame for seaborn
    long_df = summary_df.melt(
        id_vars=["condition", "sub_cond"],
        value_vars=["n_items", "total_continuations", "total_tests"],
        var_name="metric",
        value_name="count",
    )

    # Short, readable metric labels
    metric_labels = {
        "n_items": "Items",
        "total_continuations": "Continuations",
        "total_tests": "Tests",
    }
    long_df["metric"] = long_df["metric"].map(metric_labels)

    # Sort subconditions in a stable order
    long_df = long_df.sort_values(["condition", "sub_cond", "metric"])

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        data=long_df,
        x="sub_cond",
        y="count",
        hue="metric",
    )
    ax.set_xlabel("Subcondition", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Thesis Subconditions: Items, Continuations, and Tests per Subcondition",
        fontsize=13,
        pad=12,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved subcondition count plot to: {plot_path}")


def main() -> None:
    # Assume script is under scripts/dataset_validation/
    project_root = Path(__file__).resolve().parents[2]

    df = load_thesis_subconditions_csv(project_root)
    summary_df = aggregate_by_subcondition(df)

    print_summary_table(summary_df)

    results_dir = project_root / "results" / "dataset_validation"
    plot_subcondition_counts(summary_df, results_dir)


if __name__ == "__main__":
    main()

