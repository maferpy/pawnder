import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from pipelines.text import process_text_pipeline
from pipelines.image import process_image_pipeline
from pipelines.tabular import process_tabular_pipeline
from utils.inference import predict_adoption_time, generate_adoption_tips, load_all
import io

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="🐶 Pawnder 🐱", layout="wide")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ------------------------
# SESSION STATE
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "recomendaciones"  # default


# ------------------------
# ESTILOS
# ------------------------
st.markdown("""
<style>
/* Contenedor tipo header */
.header-container {
    background: linear-gradient(90deg, #D6653E, #E87C8D);
    padding: 20px 30px;
    border-radius: 20px;
    margin-bottom: 20px;
}

/* Botones */
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

/* Centrar imagen */
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
        for k in ["ranked","df_f","pos","selected_pet"]:
            st.session_state.pop(k, None)

# 🔹 LOGO CENTRO
with col_logo:
    st.markdown('<div class="logo-center">', unsafe_allow_html=True)
    st.image("img/PAWNDER_img.png", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

# 🔹 BOTÓN DERECHA
with col2:
    if st.button("🐱 Poner en adopción", use_container_width=True):
        st.session_state.page = "adopcion"
        for k in ["adopcion_form"]:
            st.session_state.pop(k, None)

st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
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
# CARGAR MODELOS Y DATOS
# ------------------------
df, embeddings = load_data()
model = load_model()
df = map_images("datasets/train_images", df)
df["activity_level"] = df["Description"].apply(infer_activity)

# ------------------------
# PÁGINA RECOMENDACIONES
# ------------------------
if st.session_state.page == "recomendaciones":
    st.title("🐶 Pawnder - Encuentra tu mascota ideal")

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

    run = st.button("🐾 Recomiéndame!")
    if run:
        user_text = f"Tipo: {pref_type}, Edad: {pref_age}, Actividad: {activity}, Tiene niños: {has_kids}, Tiene mascotas: {has_pets}, Tiempo: {time_available}, Hogar: {home_type}, Preferencias: {other_pref}"
        user_emb = model.encode([user_text])
        df_f = df.copy()
        df_f = df_f[df_f["Type"] == (1 if pref_type=="Perro" else 2)]
        df_f = df_f[(df_f["Age"] >= pref_age[0]) & (df_f["Age"] <= pref_age[1])]
        if len(df_f) < 5: df_f = df.copy()
        idx_original = df_f.index.tolist()
        emb_f = embeddings[idx_original]
        df_f = df_f.reset_index(drop=True)

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

        if st.session_state.pos+3 < len(st.session_state.ranked):
            if st.button("Ver más 🐾"): st.session_state.pos+=3; st.rerun()

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
                    {
                        "role": "system",
                        "content": "Eres un narrador. Responde solo con la historia, sin introducciones."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )

        st.info(resp.choices[0].message.content)

# ------------------------
# PÁGINA ADOPCIÓN
# ------------------------
elif st.session_state.page == "adopcion":

    st.title("🐶 Pawnder 🐱")
    st.write("Conoce en cuánto tiempo será adoptada tu mascota")

    col_img, col_inputs = st.columns([1, 2])

    with col_img:
        uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=300)

        Type = st.selectbox("Tipo", ["🐶 Perro","🐱 Gato"])
        Age = st.number_input("Edad (meses)", 0, 200, 12)
        Gender = st.selectbox("Género", ["♂️ Macho","♀️ Hembra","⚖️ Mixto"])
        Breed1 = st.text_input("Raza")

        MaturitySize = st.selectbox("Tamaño", ["Pequeño","Mediano","Grande","Extra Grande"])
        FurLength = st.selectbox("Pelaje", ["Corto","Medio","Largo"])
        Fee = st.number_input("💲 Adopción", 0, 1000, 0)

    with col_inputs:
        description = st.text_area("Descripción")

        Color1 = st.selectbox("Color1", ["Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])
        Color2 = st.selectbox("Color2", ["Ninguno","Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])
        Color3 = st.selectbox("Color3", ["Ninguno","Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])

        Vaccinated = st.selectbox("Vacunado", ["✅ Sí","❌ No","❓ No estoy seguro"])
        Dewormed = st.selectbox("Desparasitado", ["✅ Sí","❌ No","❓ No estoy seguro"])
        Sterilized = st.selectbox("Esterilizado", ["✅ Sí","❌ No","❓ No estoy seguro"])
        Health = st.selectbox("Salud", ["💚 Saludable","⚠️ Lesión menor","❌ Lesión grave"])

    if st.button("🔮 Descubre su tiempo de adopción!"):
        if st.session_state.uploaded_bytes is None or description.strip() == "":
            st.error("Sube imagen y descripción")
            st.stop()

    # 🔥 CARGA MODELOS SOLO AQUÍ
        if "model_stack" not in st.session_state:
            with st.spinner("Cargando modelos..."):
                model, pca, fee_cfg = load_all()
                st.session_state.model_stack = model
                st.session_state.pca = pca
                st.session_state.fee_cfg = fee_cfg

        # MAPS
        type_map = {"🐶 Perro":1,"🐱 Gato":2}
        gender_map = {"♂️ Macho":1,"♀️ Hembra":2,"⚖️ Mixto":3}
        maturity_map = {"Pequeño":1,"Mediano":2,"Grande":3,"Extra Grande":4}
        fur_map = {"Corto":1,"Medio":2,"Largo":3}
        yes_no_map = {"✅ Sí":1,"❌ No":2,"❓ No estoy seguro":3}
        health_map = {"💚 Saludable":1,"⚠️ Lesión menor":2,"❌ Lesión grave":3}
        color_map = {"Ninguno":0,"Negro":1,"Marrón":2,"Dorado":3,"Amarillo":4,"Crema":5,"Gris":6,"Blanco":7}

        user_input = {
            "Type": type_map[Type],
            "Age": Age,
            "Gender": gender_map[Gender],
            "Breed1": Breed1,
            "Color1": color_map[Color1],
            "Color2": color_map[Color2],
            "Color3": color_map[Color3],
            "MaturitySize": maturity_map[MaturitySize],
            "FurLength": fur_map[FurLength],
            "Vaccinated": yes_no_map[Vaccinated],
            "Dewormed": yes_no_map[Dewormed],
            "Sterilized": yes_no_map[Sterilized],
            "Health": health_map[Health],
            "Quantity": 1,
            "VideoAmt": 0,
            "PhotoAmt": 1,
            "Fee": Fee,
            "Description": description
        }
        try:
            image_bytes = uploaded_file.read()
            
            prediction = predict_adoption_time(
                user_input,
    image_bytes,
    st.session_state.model_stack,
    st.session_state.pca,
    st.session_state.fee_cfg
)
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            st.stop()

        labels = {
            0:"⏳ Adopción rápida",
            1:"⏳ Media",
            2:"⏳ Lenta",
            3:"⏳ Muy lenta",
            4:"❌ Sin adopción"
        }
        

        st.success(labels[int(prediction)])

        tips = generate_adoption_tips(
            description,
            type_map[Type],
            Age,
            maturity_map[MaturitySize],
            Fee,
            prediction
        )

        with st.expander("💡 Consejos"):
            st.write(tips)