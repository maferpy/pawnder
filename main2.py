import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------
# CLIENTE OPENAI
# ------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# FUNCIONES REUTILIZABLES
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/df_w_text_e5base_final.csv")
    embeddings = np.load("embeddings/embeddings_text_t.npy")
    return df, embeddings

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_data
def map_images(folder, df):
    df = df.copy()
    files = os.listdir(folder)
    mapping = {}
    for f in files:
        if f.lower().endswith(".jpg"):
            pid = f.split("-")[0]
            mapping.setdefault(pid, []).append(os.path.join(folder, f))
    df["image_paths"] = df["PetID"].astype(str).map(mapping)
    df["image_paths"] = df["image_paths"].apply(lambda x: x if isinstance(x, list) else [])
    return df

def infer_activity(desc):
    d = str(desc).lower()
    if any(w in d for w in ["energetic", "hyper", "very active"]):
        return "Alto"
    elif any(w in d for w in ["playful", "active"]):
        return "Medio"
    elif any(w in d for w in ["calm", "quiet", "lazy"]):
        return "Bajo"
    return "Medio"

@st.cache_data
def translate_text(text, client=client):
    """
    Traduce el texto al español usando OpenAI.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Traduce al español de forma natural."},
                {"role": "user", "content": str(text)}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except:
        return text

# ------------------------
# INICIALIZACIÓN
# ------------------------
@st.cache_resource
def init_app():
    df, embeddings = load_data()
    df = map_images("datasets/train_images", df)
    df["activity_level"] = df["Description"].apply(infer_activity)
    return df, embeddings

# Session state
if "pos" not in st.session_state: st.session_state.pos = 0
if "ranked" not in st.session_state: st.session_state.ranked = None
if "df_f" not in st.session_state: st.session_state.df_f = None
if "selected_pet" not in st.session_state: st.session_state.selected_pet = None

# ------------------------
# PÁGINA RECOMENDACIONES
# ------------------------
def page_recommendations(df, client):
    if st.session_state.page != "recomendaciones":
        return

    st.title("🐾 Pawnder - Encuentra tu mascota ideal")

    # ------------------------
    # FILTROS
    # ------------------------
    col1, col2 = st.columns(2)

    with col1:
        pref_type = st.radio("Tipo", ["Perro", "Gato"])
        pref_age = st.slider("Edad (meses)", 0, 200, (0, 24))
        activity = st.selectbox("Actividad", ["Bajo", "Medio", "Alto"])
        home_type = st.selectbox("🏠 Hogar", ["Departamento", "Casa con patio"])

    with col2:
        has_kids = st.selectbox("👦🏻 Niños", ["Sí", "No"])
        has_pets = st.selectbox("🐶 Mascotas", ["Sí", "No"])
        time_available = st.selectbox("⏰ Tiempo", ["Poco", "Medio", "Mucho"])
        health_pref = st.selectbox("🩺 Salud", ["Cualquiera", "Saludable", "Especial"])

    max_fee = st.slider("💸 Costo máximo", 0, 1000, 200)
    priority = st.selectbox("🚨 ¿Casos urgentes?", ["No importa", "Sí"])

    # ------------------------
    # BOTÓN
    # ------------------------
    if st.button("🐾 Recomiéndame!", key="recomienda"):

        df_f = df.copy()

        # Tipo
        df_f = df_f[df_f["Type"] == (1 if pref_type=="Perro" else 2)]

        # Edad
        df_f = df_f[(df_f["Age"] >= pref_age[0]) & (df_f["Age"] <= pref_age[1])]

        # Actividad (flexible según tiempo)
        if time_available == "Poco":
            df_f = df_f[df_f["activity_level"] == "Bajo"]
        elif time_available == "Medio":
            df_f = df_f[df_f["activity_level"].isin(["Bajo","Medio"])]
        else:
            pass  # Mucho → cualquiera

        # Niños
        if has_kids == "Sí":
            df_f = df_f[df_f["good_with_kids"] == 1]

        # Mascotas
        if has_pets == "Sí":
            df_f = df_f[df_f["good_with_pets"] == 1]

        # Hogar
        if home_type == "Departamento":
            df_f = df_f[df_f["MaturitySize"].isin([1,2])]

        # Salud
        if health_pref == "Saludable":
            df_f = df_f[df_f["has_health_issue"] == 0]
        elif health_pref == "Especial":
            df_f = df_f[df_f["has_health_issue"] == 1]

        # Costo
        df_f = df_f[df_f["Fee"] <= max_fee]

        # Fallback si hay pocos resultados
        if len(df_f) < 5:
            df_f = df.copy()

        # ------------------------
        # SCORING INTELIGENTE
        # ------------------------
        score = np.zeros(len(df_f))

        score += (df_f["activity_level"] == activity) * 2

        if has_kids == "Sí":
            score += df_f["good_with_kids"] * 2

        if has_pets == "Sí":
            score += df_f["good_with_pets"] * 2

        score += df_f["is_friendly"] * 1
        score += df_f["urgent"] * 2

        df_f["score"] = score
        df_f = df_f.sort_values("score", ascending=False)

        st.session_state.df_f = df_f.reset_index(drop=True)
        st.session_state.pos = 0

    # ------------------------
    # MOSTRAR RESULTADOS
    # ------------------------
    if "df_f" in st.session_state and not st.session_state.df_f.empty:

        df_f = st.session_state.df_f
        start = st.session_state.pos
        end = start + 3

        for _, row in df_f.iloc[start:end].iterrows():

            tipo = "🐶 Perro" if row["Type"]==1 else "🐱 Gato"
            st.markdown(f"### {row['Name']} - {tipo}")

            # 💡 EXPLICACIÓN
            razones = []
            if row["activity_level"] == activity:
                razones.append("tu nivel de actividad")
            if has_kids=="Sí" and row["good_with_kids"]==1:
                razones.append("es bueno con niños")
            if has_pets=="Sí" and row["good_with_pets"]==1:
                razones.append("convive con otras mascotas")
            if row["urgent"]==1:
                razones.append("necesita adopción urgente")

            if razones:
                st.success(f"💡 Recomendado porque coincide con {', '.join(razones)}")

            # Imagen
            if row["image_paths"]:
                st.image(row["image_paths"][0], width=200)

            st.markdown(f"""
            - Edad: {row['Age']} meses  
            - Actividad: {row['activity_level']}  
            - Costo: ${row['Fee']}  
            """)

            # Botón historia
            if st.button(f"✨ Historia con {row['Name']}", key=f"hist_{row['PetID']}"):
                st.session_state.selected_pet = row["PetID"]

        # Ver más
        if end < len(df_f):
            if st.button("Ver más 🐾"):
                st.session_state.pos += 3

    # ------------------------
    # HISTORIA
    # ------------------------
    if "selected_pet" in st.session_state and st.session_state.selected_pet is not None:

        row = st.session_state.df_f[
            st.session_state.df_f["PetID"] == st.session_state.selected_pet
        ].iloc[0]

        prompt = f"""
        Usuario con {has_kids} niños, {has_pets} mascotas,
        vive en {home_type} y tiene tiempo {time_available}.
        Mascota: {row['Description']}.
        Cuenta una historia corta y emotiva.
        """

        with st.spinner("Generando historia..."):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content":"Eres un narrador cálido."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.7
            )

        st.info(resp.choices[0].message.content)

    # ------------------------
    # VOLVER
    # ------------------------
    if st.button("⬅️ Volver al menú", key="volver_menu"):
        st.session_state.page = "home"