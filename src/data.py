import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing


def load_classification_data():
    """
    Încarcă dataset-ul Breast Cancer Wisconsin pentru problema de clasificare.
    Target:
        0 = malignant
        1 = benign
    """
    data = load_breast_cancer(as_frame=True)

    X = data.data
    y = data.target

    df = X.copy()
    df["target"] = y

    target_names = dict(enumerate(data.target_names))

    return df, X, y, target_names


def load_regression_data():
    """
    Încarcă dataset-ul California Housing pentru problema de regresie.
    Target:
        MedHouseVal = valoarea mediană a locuințelor, exprimată în sute de mii de dolari.
    """
    data = fetch_california_housing(as_frame=True)

    X = data.data
    y = data.target

    df = X.copy()
    df["MedHouseVal"] = y

    return df, X, y