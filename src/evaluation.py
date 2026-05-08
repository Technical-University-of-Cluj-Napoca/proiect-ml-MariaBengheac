import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluează un model de clasificare binară folosind metricile cerute:
    accuracy, precision, recall, F1, ROC-AUC și matricea de confuzie.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["ROC-AUC"] = np.nan

    metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)

    return metrics


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluează un model de regresie folosind metricile cerute:
    MSE, MAE, RMSE și R2.
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "MSE": mse,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, y_pred),
    }

    return metrics


def classification_results_to_dataframe(results):
    """
    Transformă rezultatele modelelor de clasificare într-un DataFrame sortat
    descrescător după F1 Score.
    """

    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "ROC-AUC": metrics["ROC-AUC"],
        })

    return pd.DataFrame(rows).sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)


def regression_results_to_dataframe(results):
    """
    Transformă rezultatele modelelor de regresie într-un DataFrame sortat
    descrescător după R2.
    """

    rows = []

    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "MSE": metrics["MSE"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        })

    return pd.DataFrame(rows).sort_values(
        by="R2",
        ascending=False
    ).reset_index(drop=True)