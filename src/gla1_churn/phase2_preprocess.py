
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "reports"

for directory in [PROCESSED_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Convert accumulated charges to numeric.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    missing_charges = df["TotalCharges"].isna()
    if not df.loc[missing_charges, "tenure"].eq(0).all():
        raise ValueError(
            "Unexpected missing TotalCharges outside zero-tenure records."
        )

    # Domain-based cleaning: no recorded tenure means no accumulated
    # tenure-based charges recorded in these 11 rows.
    df.loc[missing_charges, "TotalCharges"] = 0.0

    # customerID identifies a customer; it is not a predictive feature.
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")

    numeric_features = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    # Keep the test set separate until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # Fit preprocessing on training data only.
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    def dense_array(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    train_df = pd.DataFrame(
        dense_array(X_train_transformed), columns=feature_names
    )
    test_df = pd.DataFrame(
        dense_array(X_test_transformed), columns=feature_names
    )

    train_df.to_csv(
        PROCESSED_DIR / "X_train_transformed.csv", index=False
    )
    test_df.to_csv(
        PROCESSED_DIR / "X_test_transformed.csv", index=False
    )
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False, header=["Churn"])
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False, header=["Churn"])

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # Detect potential outliers using IQR on training data only.
    # Values are retained because extreme tenure/charges can be genuine.
    outlier_report = {}
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        values = X_train[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = (values < lower) | (values > upper)

        outlier_report[column] = {
            "lower_iqr_bound": round(float(lower), 3),
            "upper_iqr_bound": round(float(upper), 3),
            "flagged_count": int(flagged.sum()),
            "treatment": "Retained; robust scaling used",
        }

    report = {
        "original_rows": int(len(df)),
        "missing_total_charges_filled_with_zero": int(missing_charges.sum()),
        "customerID_used_as_feature": False,
        "target_mapping": {"No": 0, "Yes": 1},
        "split": {
            "method": "train_test_split with stratify=y",
            "test_size": 0.20,
            "random_state": 42,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_churn_counts": {
                str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "test_churn_counts": {
                str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
            },
        },
        "feature_counts": {
            "numeric_input_features": len(numeric_features),
            "categorical_input_features": len(categorical_features),
            "transformed_features": len(feature_names),
        },
        "outlier_analysis": outlier_report,
        "leakage_prevention": (
            "Preprocessor fitted only on X_train; X_test was transformed only."
        ),
    }

    (TABLES_DIR / "phase2_preprocessing_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("PHASE 2 PREPROCESSING COMPLETE")
    print("=" * 40)
    print(f"Original rows: {len(df)}")
    print(f"Missing TotalCharges filled: {int(missing_charges.sum())}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Transformed features: {len(feature_names)}")
    print("\nTraining target counts:")
    print(y_train.value_counts().sort_index().rename(index={0: "No", 1: "Yes"}))
    print("\nTesting target counts:")
    print(y_test.value_counts().sort_index().rename(index={0: "No", 1: "Yes"}))
    print("\nPotential outliers:")
    print(json.dumps(outlier_report, indent=2))
    print("\nSaved transformed datasets to:", PROCESSED_DIR)
    print("Saved preprocessor to:", MODELS_DIR / "preprocessor.joblib")
    print("Saved summary to:", TABLES_DIR / "phase2_preprocessing_summary.json")


if __name__ == "__main__":
    main()
