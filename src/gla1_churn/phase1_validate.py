
from pathlib import Path
import json

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
REPORTS_DIR = ROOT / "reports"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    # Load the raw dataset without modifying the source file.
    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "customerID", "tenure", "MonthlyCharges",
        "TotalCharges", "Churn"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Required columns missing: {sorted(missing_columns)}")

    # Normalize whitespace-only cells and convert TotalCharges to numeric.
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    missing_counts = df.isna().sum()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()

    churn_counts = df["Churn"].value_counts(dropna=False)
    churn_percentages = (
        df["Churn"].value_counts(normalize=True, dropna=False) * 100
    )

    # Inspect numeric data and categorical values.
    numeric_summary = df[numeric_columns].describe().round(3)
    categorical_summary = {
        column: {
            "unique_values": int(df[column].nunique(dropna=True)),
            "top_values": {
                str(key): int(value)
                for key, value in df[column].value_counts(
                    dropna=False
                ).head(10).items()
            },
        }
        for column in categorical_columns
    }

    zero_tenure = df[df["tenure"].eq(0)]
    blank_total_charges = df["TotalCharges"].isna()

    # Exact duplicate rows and duplicate customer identifiers.
    exact_duplicates = int(df.duplicated().sum())
    duplicate_customer_ids = int(
        df["customerID"].duplicated().sum()
    )

    # Potential numeric outliers using the 1.5 x IQR rule.
    outlier_counts = {}
    for column in numeric_columns:
        values = df[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[column] = {
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
            "count": int(((values < lower) | (values > upper)).sum()),
        }

    summary = {
        "dataset_file": str(DATA_FILE),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in missing_counts.items()
            if count > 0
        },
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_customer_ids": duplicate_customer_ids,
        "target_distribution": {
            str(label): {
                "count": int(churn_counts[label]),
                "percentage": round(float(churn_percentages[label]), 2),
            }
            for label in churn_counts.index
        },
        "numeric_summary": numeric_summary.to_dict(),
        "categorical_summary": categorical_summary,
        "zero_tenure_rows": int(zero_tenure.shape[0]),
        "missing_total_charges_rows": int(blank_total_charges.sum()),
        "missing_total_charges_and_zero_tenure": int(
            (blank_total_charges & df["tenure"].eq(0)).sum()
        ),
        "numeric_iqr_outlier_counts": outlier_counts,
    }

    json_path = TABLES_DIR / "phase1_dataset_summary.json"
    text_path = REPORTS_DIR / "phase1_validation.txt"

    json_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    report_lines = [
        "PHASE 1 — DATASET VALIDATION",
        "=" * 40,
        f"Dataset: {DATA_FILE.name}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "COLUMN DATA TYPES",
        df.dtypes.to_string(),
        "",
        "MISSING VALUES",
        missing_counts[missing_counts > 0].to_string()
        if missing_counts.any()
        else "None",
        "",
        f"Exact duplicate rows: {exact_duplicates}",
        f"Duplicate customer IDs: {duplicate_customer_ids}",
        "",
        "CHURN DISTRIBUTION",
        pd.DataFrame({
            "count": churn_counts,
            "percentage": churn_percentages.round(2),
        }).to_string(),
        "",
        "NUMERIC SUMMARY",
        numeric_summary.to_string(),
        "",
        f"Rows with tenure = 0: {len(zero_tenure)}",
        f"Rows with missing TotalCharges: {int(blank_total_charges.sum())}",
        f"Rows with both conditions: {summary['missing_total_charges_and_zero_tenure']}",
        "",
        "POTENTIAL OUTLIERS (1.5 x IQR RULE)",
        json.dumps(outlier_counts, indent=2),
        "",
        "Note: IQR flags indicate potential outliers, not automatic errors.",
        "No rows were deleted or imputed in this validation step.",
    ]

    text_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nPHASE 1 VALIDATION COMPLETE")
    print("=" * 40)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(missing_counts.sum())} cells")
    print(f"Exact duplicate rows: {exact_duplicates}")
    print(f"Duplicate customer IDs: {duplicate_customer_ids}")
    print("\nMissing values by column:")
    print(missing_counts[missing_counts > 0].to_string()
          if missing_counts.any() else "None")
    print("\nChurn distribution:")
    print(pd.DataFrame({
        "count": churn_counts,
        "percentage": churn_percentages.round(2),
    }).to_string())
    print(f"\nSaved: {json_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
