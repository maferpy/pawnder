import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI

# ------------------------
# Inicializar sesión
# ------------------------
if "pos" not in st.session_state: st.session_state.pos = 0
if "ranked" not in st.session_state: st.session_state.ranked = None
if "df_f" not in st.session_state: st.session_state.df_f = None
if "selected_pet" not in st.session_state: st.session_state.selected_pet = None

# ------------------------
# Cliente OpenAI
# ------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------
# Inferir nivel de actividad desde la descripción
# ------------------------
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
# Cargar datos y mapear imágenes
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/df_w_text_e5base_final.csv")

    # Nivel de actividad
    df["activity_level"] = df["Description"].apply(infer_activity)

    # Mapear imágenes
    folder = "datasets/train_images"
    files = os.listdir(folder)
    mapping = {}
    for f in files:
        if f.lower().endswith(".jpg"):
            pid = f.split("-")[0]
            mapping.setdefault(pid, []).append(os.path.join(folder, f))
    df["image_paths"] = df["PetID"].astype(str).map(mapping)
    df["image_paths"] = df["image_paths"].apply(lambda x: x if isinstance(x, list) else [])

    return df

df = load_data()

# ------------------------
# Página de recomendaciones
# ------------------------
def page_recommendations(df, client):
    st.title("🐾 Pawnder - Encuentra tu mascota ideal")

    # ------------------------
    # Preferencias usuario
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
    # Reiniciar resultados si cambian los filtros
    # ------------------------
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}

    current_filters = {
        "tipo": pref_type,
        "edad": pref_age,
        "actividad": activity,
        "niños": has_kids,
        "mascotas": has_pets,
        "hogar": home_type,
        "tiempo": time_available,
        "salud": health_pref,
        "max_fee": max_fee,
        "priority": priority
    }

    if current_filters != st.session_state.last_filters:
        st.session_state.df_f = None
        st.session_state.pos = 0
        st.session_state.ranked = None
        st.session_state.last_filters = current_filters

    # ------------------------
    # Botón Recomiéndame
    # ------------------------
    if st.button("🐾 Recomiéndame!", key="recomienda"):

        df_f = df.copy()

        # Tipo
        df_f = df_f[df_f["Type"] == (1 if pref_type=="Perro" else 2)]

        # Edad
        df_f = df_f[(df_f["Age"] >= pref_age[0]) & (df_f["Age"] <= pref_age[1])]

        # Actividad según tiempo disponible
        if time_available == "Poco":
            df_f = df_f[df_f["activity_level"] == "Bajo"]
        elif time_available == "Medio":
            df_f = df_f[df_f["activity_level"].isin(["Bajo", "Medio"])]
        # Mucho → cualquiera

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
        # Prioridad urgente
        if priority == "Sí":
            df_f = df_f[df_f["urgent"] == 1]

        # Fallback si hay pocos resultados
        if len(df_f) < 5:
            df_f = df.copy()

        # ------------------------
        # Scoring inteligente
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

        # Guardar en sesión
        st.session_state.df_f = df_f.reset_index(drop=True)
        st.session_state.pos = 0
        st.session_state.ranked = st.session_state.df_f.index.to_numpy()

    # ------------------------
    # Mostrar resultados
    # ------------------------
    df_f = st.session_state.df_f
    ranked = st.session_state.ranked

    if df_f is not None and not df_f.empty and ranked is not None:
        start = st.session_state.pos
        end = start + 3

        for _, row in df_f.iloc[start:end].iterrows():
            tipo = "🐶 Perro" if row["Type"]==1 else "🐱 Gato"
            st.markdown(f"### {row['Name']} - {tipo}")

            razones = []
            if row["activity_level"] == activity: razones.append("tu nivel de actividad")
            if has_kids=="Sí" and row["good_with_kids"]==1: razones.append("es bueno con niños")
            if has_pets=="Sí" and row["good_with_pets"]==1: razones.append("convive con otras mascotas")
            if row["urgent"]==1: razones.append("necesita adopción urgente")
            if razones:
                st.success(f"💡 Recomendado porque coincide con {', '.join(razones)}")

            # ------------------------
            # Mostrar imágenes
            # ------------------------
            if "image_paths" in df_f.columns and isinstance(row["image_paths"], list):
                paths = row["image_paths"]
                cols = st.columns(min(3, len(paths)))
                for i, img in enumerate(paths[:3]):
                    cols[i].image(img, width=200)

            st.markdown(f"""
            - Edad: {row['Age']} meses  
            - Actividad: {row['activity_level']}  
            - Costo: ${row['Fee']}  
            """)

            # Botón historia
            if st.button(f"✨ Cómo sería tu vida con {row['Name']}", key=f"hist_{row['PetID']}..."):
                st.session_state.selected_pet = row["PetID"]

        # ------------------------
        # Ver más
        # ------------------------
        if end < len(df_f):
            if st.button("Ver más 🐾"):
                st.session_state.pos += 3

    else:
        st.info("No encontramos mascotas que coincidan con tus filtros. Ajusta tus preferencias.")

    # ------------------------
    # Historia con la mascota
    # ------------------------
    if "selected_pet" in st.session_state and st.session_state.selected_pet is not None:
        row = st.session_state.df_f[
            st.session_state.df_f["PetID"] == st.session_state.selected_pet
        ].iloc[0]

        prompt = f"""
        Usuario con {has_kids} niños, {has_pets} mascotas,
        vive en {home_type} y tiene tiempo {time_available}.
        Mascota: {row['Description']}.
        Cuenta una historia corta y emotiva sobre cómo sería la vida del usuario con esta mascota.
        """

        with st.spinner("Tu vida con esta mascota..."):
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
    # Volver al menú
    # ------------------------
    if st.button("⬅️ Volver al menú", key="volver_menu"):
        st.session_state.page = "home"