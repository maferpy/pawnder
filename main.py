import streamlit as st
import numpy as np
from PIL import Image
import lightgbm
print(lightgbm.__version__)
from pipelines.text import process_text_pipeline
from pipelines.image import process_image_pipeline
from pipelines.tabular import process_tabular_pipeline
from utils.inference import predict_adoption_time

def main():
    st.title("🐶 Pawnder 🐱")
    st.write("Conoce en cuánto tiempo será adoptada tu mascota")

    # -------------------
    # IMAGEN + TEXTO
    # -------------------
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
    description = st.text_area("Descripción")

    # -------------------
    # TABULAR INPUTS
    # -------------------
    Type = st.selectbox("Tipo (1=Perro, 2=Gato)", [1, 2])

    Age = st.number_input("Edad", 0, 200, 12)

    Gender = st.selectbox("Género", [1, 2, 3])

    Breed1 = st.text_input("Raza (ej. border collie)")

    Color1 = st.selectbox("Color1", [0,1,2,3,4,5])
    Color2 = st.selectbox("Color2", [0,1,2,3,4,5])
    Color3 = st.selectbox("Color3", [0,1,2,3,4,5])

    MaturitySize = st.selectbox("Tamaño", [1,2,3,4])
    FurLength = st.selectbox("Largo de pelaje", [1,2,3])

    Vaccinated = st.selectbox("Vacunado", [0,1,2])
    Dewormed = st.selectbox("Desparasitado", [0,1,2])
    Sterilized = st.selectbox("Esterilizado", [0,1,2])

    Health = st.selectbox("Salud", [1,2,3])

    Quantity = 1
    VideoAmt = 0
    PhotoAmt = 1

    Fee = st.number_input("Fee", 0, 1000, 0)

    # -------------------
    # BOTÓN
    # -------------------
    if st.button("Predecir"):

        if uploaded_file is not None and description != "":

            # 1. Se crea el input del usuario
            user_input = {
                "Type": Type,
                "Age": Age,
                "Gender": Gender,
                "Breed1": Breed1,
                "Breed2": Breed2,
                "Color1": Color1,
                "Color2": Color2,
                "Color3": Color3,
                "MaturitySize": MaturitySize,
                "FurLength": FurLength,
                "Vaccinated": Vaccinated,
                "Dewormed": Dewormed,
                "Sterilized": Sterilized,
                "Health": Health,
                "Quantity": Quantity,
                "VideoAmt": VideoAmt,
                "PhotoAmt": PhotoAmt,
                "Fee": Fee,
                "Description": description
            }

        image = Image.open(uploaded_file)

        prediction = predict_adoption_time(
            user_input,
            image
        )
        st.success(f"Predicción: {prediction}")

        else:
            st.error("Falta imagen o descripción")

if __name__ == "__main__":
    main()