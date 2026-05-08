import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data import load_classification_data

st.set_page_config(
    page_title="Clasificare",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Problema de clasificare")
st.subheader("Breast Cancer Wisconsin Diagnostic")

st.markdown(
    """
    Această pagină prezintă problema de clasificare binară folosind dataset-ul
    Breast Cancer Wisconsin Diagnostic.

    Scopul este predicția tipului de tumoră:

    `0` = malignant
    `1` = benign
    """
)


@st.cache_data
def load_data():
    return load_classification_data()


@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models" / "classification"

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
        "CatBoost": "catboost.joblib",
        "XGBoost": "xgboost.joblib",
    }

    loaded_models = {}

    for model_name, file_name in model_files.items():
        file_path = models_dir / file_name
        if file_path.exists():
            loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


@st.cache_data
def load_results():
    results_path = PROJECT_ROOT / "reports" / "tables" / "classification_tuned_results.csv"

    if results_path.exists():
        return pd.read_csv(results_path)

    return pd.DataFrame()


df, X, y, target_names = load_data()
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

    st.subheader("Distribuția claselor")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="target", ax=ax)
    ax.set_title("Distribuția claselor")
    ax.set_xlabel("Clasa")
    ax.set_ylabel("Număr de observații")
    st.pyplot(fig)

    st.subheader("Matricea de corelație")

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matricea de corelație")
    st.pyplot(fig)

with tab2:
    st.header("Selectarea modelului și realizarea predicției")

    if not models:
        st.error("Nu au fost găsite modele salvate în models/classification/.")
    else:
        selected_model_name = st.selectbox(
            "Alege modelul:",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        st.subheader("Introdu valorile caracteristicilor")

        input_values = {}

        cols = st.columns(3)

        for index, feature in enumerate(X.columns):
            col = cols[index % 3]

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

        if st.button("Prezice clasa"):
            prediction = selected_model.predict(input_df)[0]

            if hasattr(selected_model, "predict_proba"):
                probability = selected_model.predict_proba(input_df)[0]
            else:
                probability = None

            class_name = target_names[prediction]

            st.success(f"Predicție: **{class_name}**")

            if probability is not None:
                st.write("Probabilități:")
                prob_df = pd.DataFrame({
                    "Clasă": [target_names[0], target_names[1]],
                    "Probabilitate": probability
                })
                st.dataframe(prob_df)

with tab3:
    st.header("Rezultatele modelelor tunate")

    if results_df.empty:
        st.warning("Nu a fost găsit fișierul classification_tuned_results.csv.")
    else:
        st.dataframe(results_df)

        st.subheader("Compararea scorului F1")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=results_df,
            x="F1 Score",
            y="Model",
            ax=ax
        )
        ax.set_title("Scor F1 pentru modelele de clasificare")
        st.pyplot(fig)