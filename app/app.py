import streamlit as st

st.set_page_config(
    page_title="Proiect Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.title("Proiect Machine Learning")
st.subheader("Analiza comparată a modelelor de Machine Learning")

st.markdown(
    """
    Această aplicație prezintă rezultatele obținute în cadrul proiectului de Machine Learning.

    Proiectul conține două probleme principale:

    1. Clasificare: predicția tipului de tumoră folosind dataset-ul Breast Cancer Wisconsin.
    2. Regresie: predicția valorii mediane a locuințelor folosind dataset-ul California Housing.

    Pentru fiecare problemă au fost parcurse etapele:

    - analiza exploratorie a datelor;
    - antrenarea mai multor modele de bază;
    - compararea performanțelor;
    - ajustarea hiperparametrilor;
    - analiza curbelor de învățare;
    - interpretarea modelelor folosind SHAP.
    """
)

st.info("Folosește meniul din stânga pentru a naviga între pagina de clasificare și pagina de regresie.")