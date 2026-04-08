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

@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_data
def translate_text(text):
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

# ------------------------
# CARGAR DATOS Y MODELOS
# ------------------------
@st.cache_resource
def init_app():
    df, embeddings = load_data()
    model = load_model()
    df = map_images("datasets/train_images", df)
    df["activity_level"] = df["Description"].apply(infer_activity)
    return df, embeddings, model

df, embeddings, model = init_app()

# ------------------------
# MENÚ DE PÁGINAS
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "recomendaciones"
if "pos" not in st.session_state:
    st.session_state.pos = 0

# ------------------------
# PÁGINA RECOMENDACIONES
# ------------------------
def  page_recommendations():
    if st.session_state.page == "recomendaciones":
        st.title("🐶 Pawnder - Encuentra tu mascota ideal")

        # Preferencias usuario
        col1, col2 = st.columns(2)
        with col1:
            pref_type = st.radio("Tipo", ["Perro", "Gato"])
            pref_age = st.slider("Edad (meses)", 0, 200, (0, 24))
            activity = st.selectbox("Actividad deseada", ["Bajo", "Medio", "Alto"])
        with col2:
            has_kids = st.selectbox("👦🏻 ¿Tienes niños? 👧🏻", ["Sí", "No"])
            has_pets = st.selectbox("🦮 ¿Tienes mascotas?", ["Sí", "No"])
            time_available = st.selectbox("⏰ Tiempo disponible", ["Poco", "Medio", "Mucho"])

        home_type = st.selectbox("🏠 Tipo de hogar 🏡", ["Departamento", "Casa con patio"])
        other_pref = st.text_area("🔎 ¿Algo más que buscas?")

        # Botón para recomendar
        if st.button("🐾 Recomiéndame!"):
            user_text = f"Tipo: {pref_type}, Edad: {pref_age}, Actividad: {activity}, Tiene niños: {has_kids}, Tiene mascotas: {has_pets}, Tiempo: {time_available}, Hogar: {home_type}, Preferencias: {other_pref}"
            user_emb = model.encode([user_text])
            df_f = df.copy()
            df_f = df_f[df_f["Type"] == (1 if pref_type=="Perro" else 2)]
            df_f = df_f[(df_f["Age"] >= pref_age[0]) & (df_f["Age"] <= pref_age[1])]
            if len(df_f) < 5: df_f = df.copy()
            idx_original = df_f.index.tolist()
            emb_f = embeddings[idx_original]
            df_f = df_f.reset_index(drop=True)

            # Similitud + penalizaciones
            sims = cosine_similarity(user_emb, emb_f)[0]
            penalty = []
            for _, row in df_f.iterrows():
                p = 0
                if activity == "Bajo" and row.get("activity_level","")!="Bajo": p-=1
                if has_kids=="Sí" and row.get("good_with_kids",1)==0: p-=2
                if has_pets=="Sí" and row.get("good_with_pets",1)==0: p-=2
                penalty.append(p)
            final_score = sims + np.array(penalty)
            ranked = np.argsort(final_score)[::-1]

            st.session_state.ranked = ranked
            st.session_state.df_f = df_f
            st.session_state.pos = 0

        # Mostrar resultados
        if "ranked" in st.session_state:
            df_f = st.session_state.df_f
            start = st.session_state.pos
            end = start + 3
            idxs = st.session_state.ranked[start:end]
            pets = df_f.iloc[idxs]

            for _, row in pets.iterrows():
                tipo = "🐶 Perro" if row["Type"]==1 else "🐱 Gato"
                st.markdown(f"### {row['Name']} - {tipo}")
                if has_kids=="Sí" and row.get("good_with_kids",1)==0: st.warning("⚠️ Puede no ser ideal para niños")
                if has_pets=="Sí" and row.get("good_with_pets",1)==0: st.warning("⚠️ Puede no convivir bien con otras mascotas")
                razones=[]
                if row["activity_level"]==activity: razones.append("tu nivel de actividad")
                if pref_age[0]<=row["Age"]<=pref_age[1]: razones.append("la edad que buscas")
                st.success(f"💡 Te recomendamos este perro porque coincide con {', '.join(razones)}")
                desc_es = translate_text(row["Description"])
                st.write(desc_es)
                if row["image_paths"]:
                    cols = st.columns(min(3,len(row["image_paths"])))
                    for i,img in enumerate(row["image_paths"][:3]): cols[i].image(img,width=200)
                if st.button(f"✨ ¿Cómo sería tu vida con {row['Name']}?", key=row["PetID"]):
                    st.session_state.selected_pet = row["PetID"]
                st.write("---")

            # Ver más
            if st.session_state.pos+3 < len(st.session_state.ranked):
                if st.button("Ver más 🐾"):
                    st.session_state.pos += 3  # Incrementa la posición

        # Historia de la mascota
        if "selected_pet" in st.session_state:
            row = st.session_state.df_f[st.session_state.df_f["PetID"]==st.session_state.selected_pet].iloc[0]
            desc_es = translate_text(row["Description"])
            user_ctx = f"Usuario con {has_kids} niños, {has_pets} mascotas, vive en {home_type}, tiene {time_available} tiempo."
            pet_ctx = f"Mascota: {desc_es}, Actividad: {row['activity_level']}"
            prompt = f"""{user_ctx}\n{pet_ctx}\nDescribe cómo sería su vida juntos.
            INSTRUCCIONES IMPORTANTES:
            - NO uses frases como 'Claro', 'Aquí tienes', 'A continuación'
            - NO expliques lo que harás
            - Empieza directamente con la historia
            - Escribe como una narrativa natural
            - Usa un tono cálido y emocional
            - Máximo 120 palabras
            Responde SOLO con la historia."""
            with st.spinner("Generando historia..."):
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "Eres un narrador. Responde solo con la historia, sin introducciones."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
            st.info(resp.choices[0].message.content)
page_recommendations()