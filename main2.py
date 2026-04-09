import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI

# ------------------------
# Inicializar variables de sesión de forma segura
# ------------------------
st.session_state.setdefault("pos", 0)
st.session_state.setdefault("ranked", None)
st.session_state.setdefault("df_f", None)
st.session_state.setdefault("selected_pet", None)
st.session_state.setdefault("last_filters", {})

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
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "datasets", "df_w_text_e5base_final.csv")
    df = pd.read_csv(csv_path)

    df["activity_level"] = df["Description"].apply(infer_activity)

    folder = os.path.join(BASE_DIR, "datasets", "train_images")
    mapping = {}
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(".jpg"):
                pid = f.split("-")[0]
                mapping.setdefault(pid, []).append(os.path.join(folder, f))
    df["image_paths"] = df["PetID"].astype(str).map(mapping).apply(lambda x: x if isinstance(x, list) else [])
    return df

# ------------------------
# Función de filtrado
# ------------------------
def filter_df(df, filters):
    df_f = df.copy()
    # Tipo
    df_f = df_f[df_f["Type"] == (1 if filters["tipo"]=="Perro" else 2)]
    # Edad
    df_f = df_f[(df_f["Age"] >= filters["edad"][0]) & (df_f["Age"] <= filters["edad"][1])]
    # Actividad según tiempo disponible
    if filters["tiempo"] == "Poco":
        df_f = df_f[df_f["activity_level"] == "Bajo"]
    elif filters["tiempo"] == "Medio":
        df_f = df_f[df_f["activity_level"].isin(["Bajo","Medio"])]
    # Niños
    if filters["niños"] == "Sí":
        df_f = df_f[df_f["good_with_kids"] == 1]
    # Mascotas
    if filters["mascotas"] == "Sí":
        df_f = df_f[df_f["good_with_pets"] == 1]
    # Hogar
    if filters["hogar"] == "Departamento":
        df_f = df_f[df_f["MaturitySize"].isin([1,2])]
    # Salud
    if filters["salud"] == "Saludable":
        df_f = df_f[df_f["has_health_issue"] == 0]
    elif filters["salud"] == "Especial":
        df_f = df_f[df_f["has_health_issue"] == 1]
    # Costo
    df_f = df_f[df_f["Fee"] <= filters["max_fee"]]
    # Prioridad urgente
    if filters["priority"] == "Sí":
        df_f = df_f[df_f["urgent"] == 1]
    # Fallback
    if len(df_f) < 5:
        df_f = df.copy()
    return df_f

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
    # Guardar filtros actuales
    # ------------------------
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

    if current_filters != st.session_state.get("last_filters", {}):
        st.session_state["last_filters"] = current_filters
        st.session_state["pos"] = 0
        st.session_state["ranked"] = None

    # ------------------------
    # Aplicar filtrado automáticamente
    # ------------------------
    st.session_state["df_f"] = filter_df(df, current_filters)

    # ------------------------
    # Botón Recomiéndame! → solo scoring/ranking
    # ------------------------
    if st.button("🐾 Recomiéndame!", key="recomienda"):
        df_f = st.session_state["df_f"].copy()
        score = np.zeros(len(df_f))
        score += (df_f["activity_level"] == current_filters["actividad"])*2
        if current_filters["niños"]=="Sí": score += df_f["good_with_kids"]*2
        if current_filters["mascotas"]=="Sí": score += df_f["good_with_pets"]*2
        score += df_f["is_friendly"]*1
        score += df_f["urgent"]*2
        df_f["score"] = score
        df_f = df_f.sort_values("score", ascending=False)
        st.session_state["df_f"] = df_f.reset_index(drop=True)
        st.session_state["ranked"] = st.session_state["df_f"].index.to_numpy()
        st.session_state["pos"] = 0

    # ------------------------
    # Mostrar resultados
    # ------------------------
    df_f = st.session_state.get("df_f", None)
    ranked = st.session_state.get("ranked", None)
    start = st.session_state.get("pos", 0)
    end = start + 3

    if df_f is not None and not df_f.empty:
        for _, row in df_f.iloc[start:end].iterrows():
            tipo = "🐶 Perro" if row["Type"]==1 else "🐱 Gato"
            st.markdown(f"### {row['Name']} - {tipo}")

            razones = []
            if row["activity_level"] == current_filters["actividad"]: razones.append("tu nivel de actividad")
            if current_filters["niños"]=="Sí" and row["good_with_kids"]==1: razones.append("es bueno con niños")
            if current_filters["mascotas"]=="Sí" and row["good_with_pets"]==1: razones.append("convive con otras mascotas")
            if row["urgent"]==1: razones.append("necesita adopción urgente")
            if razones:
                st.success(f"💡 Recomendado porque coincide con {', '.join(razones)}")

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

            if st.button(f"✨ Cómo sería tu vida con {row['Name']}", key=f"hist_{row['PetID']}"):
                st.session_state["selected_pet"] = row["PetID"]

        if end < len(df_f):
            if st.button("Ver más 🐾"):
                st.session_state["pos"] += 3
    else:
        st.info("No encontramos mascotas que coincidan con tus filtros. Ajusta tus preferencias.")

    # ------------------------
    # Historia con la mascota
    # ------------------------
    selected_pet = st.session_state.get("selected_pet", None)
    if selected_pet is not None and df_f is not None:
        row = df_f[df_f["PetID"]==selected_pet].iloc[0]
        prompt = f"""
        Tú tienes {current_filters['niños']} niños, {current_filters['mascotas']} mascotas,
        vives en {current_filters['hogar']} y tienes tiempo {current_filters['tiempo']}.
        Mascota: {row['Description']}.
        Describe cómo sería tu vida con esta mascota.
        """
        with st.spinner("Tu vida con esta mascota..."):
            if client:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role":"system","content":"Eres un asistente emocional y cercano. Usa 'tú' y 'tu vida'. Evita tercera persona."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.7
                )
                st.info(resp.choices[0].message.content)
            else:
                st.info("La historia de la mascota no puede generarse sin API key.")

    # ------------------------
    # Volver al menú
    # ------------------------
    if st.button("⬅️ Volver al menú", key="volver_menu"):
        st.session_state["page"] = "home"
