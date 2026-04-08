import numpy as np
import joblib
import pandas as pd
import streamlit as st

from pipelines.tabular import process_tabular_pipeline
from pipelines.text import process_text_pipeline
from pipelines.image import process_image_pipeline

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import os

# Agregar openai

load_dotenv()  # 👈 esto carga el .env
api_key = st.secrets.get("OPENAI_API_KEY", None)

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("⚠️ OpenAI API key no encontrada. Las historias no se generarán.")
# Carga Modelos
@st.cache_resource
def load_all():
    model = joblib.load("models/stacking_model.pkl")
    pca = joblib.load("models/pca_200.pkl")
    fee_cfg = joblib.load("models/fee_bins_u.pkl")
    return model, pca, fee_cfg


@st.cache_resource
def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model, pca, fee_cfg = load_all()
text_model = load_text_model()


model_text = model["model_text"]
model_img = model["model_img"]
meta_model = model["meta_model"]

text_tab_cols = model["text_cols"]
img_cols = model["img_cols"]


# ------------------------
# PREDICCIÓN
# ------------------------
def predict_adoption_time(user_input: dict, uploaded_file):

    # TABULAR
    df = pd.DataFrame([user_input])
    tabular = process_tabular_pipeline(df)

    categorical_cols = ["age_bin", "fee_pets"]
    for col in categorical_cols:
        tabular[col] = tabular[col].astype("category").cat.codes

    tabular = tabular.drop(columns=["Description"], errors="ignore")
    tabular = tabular.drop(columns=["Breed1"], errors="ignore")
    tabular = tabular.drop(columns=["Age"], errors="ignore")
    print("tabular listo")
    print("Columnas tabulares:", tabular.columns.tolist())

    # TEXT
    df_text = pd.DataFrame([{"Description": user_input["Description"]}])
    text_df = process_text_pipeline(df_text)  
    text_array = text_df.to_numpy()
    print("text listo")
    print("Columnas tabulares:", text_df.columns.tolist())
    # IMAGE
    img_array = process_image_pipeline(uploaded_file)
    print("img listo")

    # CONCAT FEATURES
    X_text_tab = np.hstack([tabular.to_numpy(), text_array])

    # BASE MODELS
    pred_text = model_text.predict_proba(X_text_tab)
    pred_img = model_img.predict_proba(img_array)

    # STACKING
    X_meta = np.hstack([pred_text, pred_img])
    pred = meta_model.predict(X_meta)

    return int(pred[0])

# OpenAI
def generate_adoption_tips(description, animal_type, age, size, fee, prediction):



    # Mapear predicción

    speed_map = {
        0: "rápida",
        1: "media",
        2: "lenta",
        3: "muy lenta",
        4: "sin adopción"
    }

    adoption_speed = speed_map.get(int(prediction), "desconocida")

    # Prompt 
    prompt = f"""
    Soy un experto en adopción de mascotas.

    Información de la mascota:
    - Descripción: {description}
    - Tipo: {"Perro" if animal_type == 1 else "Gato"}
    - Edad: {age} meses
    - Tamaño: {size}
    - Precio: {fee}

    El modelo predice que la adopción será: {adoption_speed}.

    Instrucciones:
    - Si la adopción es rápida o media: refuerza lo positivo.
    - Si la adopción es lenta o muy lenta: da consejos para mejorar visibilidad, fotos, descripción o precio.
    - Si no se adoptaría: da recomendaciones más agresivas y concretas.

    Dame 3 consejos cortos, claros y accionables.
    Considera la descripción de la mascota.
    """

    # ------------------------
    # Llamada a OpenAI (multimodal)
    # ------------------------
    if client is not None:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ]
                }
            ],
            max_tokens=150
        )
        response.choices[0].message.content
    else:
        st.info("Los consejos no pueden generarse sin API key.")