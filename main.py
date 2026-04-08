import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import lightgbm
print(lightgbm.__version__)
from pipelines.text import process_text_pipeline
from pipelines.image import process_image_pipeline
from pipelines.tabular import process_tabular_pipeline
from utils.inference import predict_adoption_time, generate_adoption_tips
from openai import OpenAI
import os



def page_predictions(client):
    if st.session_state.page == "predictions":
        if st.button("⬅️ Volver al menú", key="back_menu_predictions"):
            st.session_state.page = "home"
            
        
        st.set_page_config(page_title="🐶 Pawnder 🐱", layout="wide")
        st.title("🐶 Pawnder 🐱")
        st.write("Conoce en cuánto tiempo será adoptada tu mascota")


        # ---------------------------
        # Layout columnas
        # ---------------------------
        col_img, col_inputs = st.columns([1, 2])

        with col_img:
            uploaded_file = st.file_uploader("📷 Sube una imagen de tu mascota", type=["jpg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", width=400)
            Type = st.selectbox("Tipo de mascota", ["🐶 Perro", "🐱 Gato"])
            Age = st.number_input("Edad (meses)", 0, 200, 12)
            Gender = st.selectbox("Género", ["♂️ Macho", "♀️ Hembra", "⚖️ Mixto"])
            Breed1 = st.text_input("Raza")
            MaturitySize = st.selectbox("Tamaño adulto", ["Pequeño", "Mediano", "Grande", "Extra Grande"])
            FurLength = st.selectbox("Largo de pelaje", ["Corto", "Medio", "Largo"])
            Fee = st.number_input("💲 Adopción (MXN)", 0, 1000, 0)

        with col_inputs:
            description = st.text_area("✏️ Descripción", height=150)

            Color1 = st.selectbox("Color principal", ["Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])
            Color2 = st.selectbox("Color secundario", ["Ninguno","Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])
            Color3 = st.selectbox("Color terciario", ["Ninguno","Negro","Marrón","Dorado","Amarillo","Crema","Gris","Blanco"])
            
            Vaccinated = st.selectbox("Vacunado", ["✅ Sí", "❌ No", "❓ No estoy seguro"])
            Dewormed = st.selectbox("Desparasitado", ["✅ Sí", "❌ No", "❓ No estoy seguro"])
            Sterilized = st.selectbox("Esterilizado", ["✅ Sí", "❌ No", "❓ No estoy seguro"])
            Health = st.selectbox("Salud", ["💚 Saludable", "⚠️ Lesión menor", "❌ Lesión grave"])
            

        Quantity = 1
        VideoAmt = 0
        PhotoAmt = 1

        if st.button("🔮 Descubre su tiempo de adopción!"):
            if not uploaded_file or description.strip() == "":
                st.error("Por favor sube una imagen y escribe una descripción")
                return

            # ---------------------------
            # Map inputs a números
            # ---------------------------
            type_map = {"🐶 Perro": 1, "🐱 Gato": 2}
            gender_map = {"♂️ Macho": 1, "♀️ Hembra": 2, "⚖️ Mixto": 3}
            maturity_map = {"Pequeño":1, "Mediano":2, "Grande":3, "Extra Grande":4}
            fur_map = {"Corto":1, "Medio":2, "Largo":3}
            yes_no_map = {"✅ Sí":1, "❌ No":2, "❓ No estoy seguro":3}
            health_map = {"💚 Saludable":1, "⚠️ Lesión menor":2, "❌ Lesión grave":3}
            color_map = {"Ninguno":0, "Negro":1, "Marrón":2,"Dorado":3,"Amarillo":4,"Crema":5,"Gris":6,"Blanco":7}

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
                "Quantity": Quantity,
                "VideoAmt": VideoAmt,
                "PhotoAmt": PhotoAmt,
                "Fee": Fee,
                "Description": description
            }

            # ---------------------------
            # Predicción
            # ---------------------------
            prediction = predict_adoption_time(user_input, uploaded_file)

            labels = {
                0: "⏳ Adopción rápida (0–7 días)",
                1: "⏳ Adopción media (1–7 días)",
                2: "⏳ Adopción lenta (8–30 días)",
                3: "⏳ Adopción muy lenta (30–90 días)",
                4: "⏳ Sin adopción (90+ días)"
            }

            # ---------------------------
            # Resultado con colores
            # ---------------------------
            pred_label = labels[int(prediction)]
            if int(prediction) == 0:
                st.success(f"🎉 {pred_label}")
            elif int(prediction) == 1:
                st.info(f"🕐 {pred_label}")
            elif int(prediction) == 2:
                st.warning(f"⏳ {pred_label}")
            else:
                st.error(f"❌ {pred_label}")

            # Consejos OpenAI
            tips = generate_adoption_tips(
                description,
                type_map[Type],
                Age,
                maturity_map[MaturitySize],
                Fee,
                prediction,
                client
            )
            with st.expander("💡 Consejos para acelerar la adopción"):
                st.write(tips)

page_predictions(client)
