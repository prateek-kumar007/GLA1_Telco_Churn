
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_FILE)

# Convert whitespace-only cells to missing values.
df = df.replace(r"^\s*$", pd.NA, regex=True)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

missing = df[df["TotalCharges"].isna()]

print("DATASET SHAPE:", df.shape)
print("\nMISSING TotalCharges RECORDS:", len(missing))
print("\nRELEVANT COLUMNS:")
print(
    missing[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)

print("\nTENURE SUMMARY FOR MISSING RECORDS:")
print(missing["tenure"].describe())

print("\nMISSING CHARGES WITH ZERO TENURE:")
print("Count:", int(missing["tenure"].eq(0).sum()))

print("\nALL ZERO-TENURE RECORDS:")
zero_tenure = df[df["tenure"].eq(0)]
print("Count:", len(zero_tenure))
print(
    zero_tenure[
        ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)
