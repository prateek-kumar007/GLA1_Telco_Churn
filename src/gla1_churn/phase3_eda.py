
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA_FILE)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Apply the documented zero-tenure assumption for EDA only.
    missing = df["TotalCharges"].isna()
    if not df.loc[missing, "tenure"].eq(0).all():
        raise ValueError("Unexpected missing charges outside zero-tenure records.")
    df.loc[missing, "TotalCharges"] = 0.0

    numeric_columns = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]

    # 1. Descriptive statistics: mean, variance, skewness, kurtosis,
    # and selected percentiles.
    stats = df[numeric_columns].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
    ).T

    stats["variance"] = df[numeric_columns].var()
    stats["skewness"] = df[numeric_columns].skew()
    # pandas kurtosis is excess kurtosis (normal distribution = 0).
    stats["excess_kurtosis"] = df[numeric_columns].kurtosis()

    stats.to_csv(TABLES_DIR / "phase3_numeric_statistics.csv")

    # 2. Churn distribution.
    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    churn_percent = df["Churn"].value_counts(normalize=True).reindex(
        ["No", "Yes"]
    ) * 100

    churn_summary = pd.DataFrame({
        "count": churn_counts,
        "percentage": churn_percent.round(2),
    })
    churn_summary.to_csv(TABLES_DIR / "phase3_churn_distribution.csv")

    # Figure 1: tenure distribution.
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="tenure", bins=30, kde=True)
    plt.title("Customer Tenure Distribution")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_tenure_histogram.png", dpi=300)
    plt.close()

    # Figure 2: monthly charges by churn status.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=["No", "Yes"])
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_monthly_charges_boxplot.png", dpi=300)
    plt.close()

    # Figure 3: tenure vs total charges, colored by churn.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="tenure",
        y="TotalCharges",
        hue="Churn",
        hue_order=["No", "Yes"],
        alpha=0.55,
        s=22,
    )
    plt.title("Tenure vs Total Charges by Churn Status")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Total Charges")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_tenure_totalcharges_scatter.png", dpi=300)
    plt.close()

    # Figure 4: numeric-feature correlation heatmap.
    plt.figure(figsize=(7, 5))
    corr = df[numeric_columns].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase3_correlation_heatmap.png", dpi=300)
    plt.close()

    # Churn rates by contract type.
    contract_churn = pd.crosstab(
        df["Contract"], df["Churn"], normalize="index"
    ) * 100
    contract_churn.to_csv(TABLES_DIR / "phase3_churn_by_contract.csv")

    # Churn rates by internet service.
    internet_churn = pd.crosstab(
        df["InternetService"], df["Churn"], normalize="index"
    ) * 100
    internet_churn.to_csv(TABLES_DIR / "phase3_churn_by_internet_service.csv")

    print("PHASE 3 EDA COMPLETE")
    print("=" * 40)
    print("\nNUMERIC STATISTICS")
    print(stats.round(3).to_string())
    print("\nCHURN DISTRIBUTION")
    print(churn_summary.to_string())
    print("\nCHURN RATE BY CONTRACT TYPE (%)")
    print(contract_churn.round(2).to_string())
    print("\nCHURN RATE BY INTERNET SERVICE (%)")
    print(internet_churn.round(2).to_string())
    print("\nSaved tables to:", TABLES_DIR)
    print("Saved figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
