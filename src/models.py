from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


def get_classification_models(random_state=42):
    """
    Returnează modelele de clasificare cerute în proiect.
    Modelele sunt inițializate cu hiperparametrii de bază/default.
    """

    models = {
        "Naive Bayes": GaussianNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=random_state
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            random_state=random_state
        ),

        "Support Vector Machine": SVC(
            probability=True,
            random_state=random_state
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state
        ),

        "CatBoost": CatBoostClassifier(
            verbose=0,
            random_state=random_state
        ),

        "Explainable Boosting Machine": ExplainableBoostingClassifier(
            random_state=random_state
        ),
    }

    return models


def get_regression_models(random_state=42):
    """
    Returnează modelele de regresie cerute în proiect.
    Modelele sunt inițializate cu hiperparametrii de bază/default.
    """

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=random_state
        ),

        "Random Forest Regressor": RandomForestRegressor(
            random_state=random_state
        ),

        "Support Vector Regressor": SVR(),

        "K-Nearest Neighbor Regressor": KNeighborsRegressor(),

        "Gaussian Process Regressor": GaussianProcessRegressor(),

        "XGBoost Regressor": XGBRegressor(
            random_state=random_state
        ),

        "CatBoost Regressor": CatBoostRegressor(
            verbose=0,
            random_state=random_state
        ),

        "Explainable Boosting Regressor": ExplainableBoostingRegressor(
            random_state=random_state
        ),
    }

    return models