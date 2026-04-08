import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "recomendaciones"
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

if st.session_state.page == "recomendaciones":
    from main2 import page_recommendations
    page_recommendations()
elif st.session_state.page == "predicciones":
    from main import page_predictions
    page_predictions()