import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data import load_regression_data


st.set_page_config(
    page_title="Regresie",
    page_icon="🏠",
    layout="wide"
)

st.title(" Problema de regresie")
st.subheader("California Housing")

st.markdown(
    """
    Această pagină prezintă problema de regresie folosind dataset-ul
    California Housing.

    Scopul este predicția valorii mediane a locuințelor dintr-o zonă,
    pe baza unor caracteristici socio-economice și geografice.

    Variabila țintă este `MedHouseVal`, exprimată în sute de mii de dolari.
    """
)


@st.cache_data
def load_data():
    return load_regression_data()


@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models" / "regression"

    model_files = {
        "CatBoost Regressor": "catboost_regressor.joblib",
        "Explainable Boosting Regressor": "explainable_boosting_regressor.joblib",
        "XGBoost Regressor": "xgboost_regressor.joblib",
        "Random Forest Regressor": "random_forest_regressor.joblib",
        "Support Vector Regressor": "support_vector_regressor.joblib",
    }

    loaded_models = {}

    for model_name, file_name in model_files.items():
        file_path = models_dir / file_name
        if file_path.exists():
            loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


@st.cache_data
def load_results():
    results_path = PROJECT_ROOT / "reports" / "tables" / "regression_tuned_results.csv"

    if results_path.exists():
        return pd.read_csv(results_path)

    return pd.DataFrame()


df, X, y = load_data()
models = load_models()
results_df = load_results()

tab1, tab2, tab3 = st.tabs([
    "Date și EDA",
    "Modele și predicție",
    "Rezultate"
])


with tab1:
    st.header("Prezentarea dataset-ului")

    st.write("Dimensiune dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Distribuția variabilei țintă")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["MedHouseVal"], kde=True, ax=ax)
    ax.set_title("Distribuția variabilei țintă MedHouseVal")
    ax.set_xlabel("MedHouseVal")
    ax.set_ylabel("Frecvență")
    st.pyplot(fig)

    st.subheader("Matricea de corelație")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matricea de corelație")
    st.pyplot(fig)


with tab2:
    st.header("Selectarea modelului și realizarea predicției")

    if not models:
        st.error("Nu au fost găsite modele salvate în models/regression/.")
    else:
        selected_model_name = st.selectbox(
            "Alege modelul:",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        st.subheader("Introdu valorile caracteristicilor")

        input_values = {}

        cols = st.columns(2)

        for index, feature in enumerate(X.columns):
            col = cols[index % 2]

            with col:
                input_values[feature] = st.number_input(
                    label=feature,
                    value=float(X[feature].mean()),
                    min_value=float(X[feature].min()),
                    max_value=float(X[feature].max())
                )

        input_df = pd.DataFrame([input_values])

        st.subheader("Date introduse")
        st.dataframe(input_df)

        if st.button("Prezice valoarea locuinței"):
            prediction = selected_model.predict(input_df)[0]

            st.success(
                f"Valoarea prezisă este: **{prediction:.3f}** sute de mii de dolari"
            )

            st.info(
                f"Aproximativ: **{prediction * 100000:,.0f} dolari**"
            )


with tab3:
    st.header("Rezultatele modelelor tunate")

    if results_df.empty:
        st.warning("Nu a fost găsit fișierul regression_tuned_results.csv.")
    else:
        st.dataframe(results_df)

        st.subheader("Compararea scorului R²")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=results_df,
            x="R2",
            y="Model",
            ax=ax
        )
        ax.set_title("Scor R² pentru modelele de regresie")
        st.pyplot(fig)

        st.subheader("Compararea RMSE")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=results_df,
            x="RMSE",
            y="Model",
            ax=ax
        )
        ax.set_title("RMSE pentru modelele de regresie")
        st.pyplot(fig)