import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "home"
from main import page_predictions
from main2 import load_data, page_recommendations,df
from openai import OpenAI
import os
import pandas as pd

if "df_f" not in st.session_state or st.session_state.df_f is None:
    st.session_state.df_f = pd.DataFrame()

df = load_data()
api_key = st.secrets.get("OPENAI_API_KEY", None)
st.write("API KEY:", api_key[:5] if api_key else "NO KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("⚠️ OpenAI API key no encontrada. Las historias no se generarán.")
# ------------------------
# Inicializar sesión
# ------------------------
st.set_page_config(page_title="🐶 Pawnder 🐱", layout="wide")

# ------------------------
# Función HOME (MENÚ)
# ------------------------
def page_home():
    # ------------------------
    # ESTILOS CSS
    # ------------------------
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(90deg, #D6653E, #E87C8D);
        padding: 20px 30px;
        border-radius: 20px;
        margin-bottom: 20px;
    }
    div.stButton > button {
        background-color: white !important;
        color: #D6653E !important;
        font-size: 16px;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ffe6e0 !important;
        color: #E87C8D !important;
    }
    .logo-center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------
    # HEADER
    # ------------------------
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    col1, col_logo, col2 = st.columns([1, 0.5, 1])

    # 🔹 BOTÓN IZQUIERDA
    with col1:
        if st.button("🐶 Recomendar mascotas", use_container_width=True):
            st.session_state.page = "recomendaciones"

    # 🔹 LOGO CENTRO
    with col_logo:
        st.markdown('<div class="logo-center">', unsafe_allow_html=True)
        st.image("img/PAWNDER_img.png", width=650)
        st.markdown('</div>', unsafe_allow_html=True)

    # 🔹 BOTÓN DERECHA
    with col2:
        if st.button("🐱 Poner en adopción", use_container_width=True):
            st.session_state.page = "predictions"

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# NAVEGACIÓN PRINCIPAL
# ------------------------
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "recomendaciones":
    page_recommendations(df, client)
elif st.session_state.page == "predictions":
    page_predictions()
