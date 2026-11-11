
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "outputs" / "models"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"

for directory in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def evaluate(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_yes": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_yes": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_yes": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train_transformed.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_transformed.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Churn"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Churn"]

    # Logistic Regression baseline.
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    model_metrics = evaluate(y_test, predictions, probabilities)

    # Majority-class baseline for comparison.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)
    dummy_probabilities = dummy.predict_proba(X_test)[:, 1]
    dummy_metrics = evaluate(y_test, dummy_predictions, dummy_probabilities)

    results = {
        "model": "Logistic Regression",
        "positive_class": "Yes (churn)",
        "test_rows": int(len(y_test)),
        "metrics": model_metrics,
        "majority_baseline_metrics": dummy_metrics,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    (TABLES_DIR / "phase4_model_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Save confusion matrix.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No churn", "Actual Churn"],
        columns=["Predicted No churn", "Predicted Churn"],
    )
    cm_df.to_csv(TABLES_DIR / "phase4_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Logistic Regression — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["No churn", "Churn"])
    ax.set_yticks([0, 1], ["No churn", "Churn"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Save ROC curve.
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Logistic Regression — ROC Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase4_roc_curve.png", dpi=300)
    plt.close(fig)

    # Save test predictions for later inspection.
    pd.DataFrame({
        "actual_churn": y_test,
        "predicted_churn": predictions,
        "predicted_churn_probability": probabilities,
    }).to_csv(PREDICTIONS_DIR / "phase4_test_predictions.csv", index=False)

    joblib.dump(model, MODELS_DIR / "logistic_regression.joblib")

    print("PHASE 4 MODEL EVALUATION COMPLETE")
    print("=" * 45)
    print("\nLOGISTIC REGRESSION")
    for metric, value in model_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nMAJORITY-CLASS BASELINE")
    for metric, value in dummy_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        zero_division=0,
    ))

    print("CONFUSION MATRIX")
    print(cm_df.to_string())

    print("\nSaved metrics:", TABLES_DIR / "phase4_model_metrics.json")
    print("Saved confusion matrix:", FIGURES_DIR / "phase4_confusion_matrix.png")
    print("Saved ROC curve:", FIGURES_DIR / "phase4_roc_curve.png")
    print("Saved model:", MODELS_DIR / "logistic_regression.joblib")


if __name__ == "__main__":
    main()
