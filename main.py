import streamlit as st
import numpy as np
from PIL import Image

from pipelines.image import extract_image_features
from pipelines.text import extract_text_features
from utils.inference import predict_adoption_time

def main():

    st.title("🐶 Pawnder 🐱")
    st.write("Predice cuánto tiempo tardará una mascota en ser adoptada")

    # INPUTS
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
    text = st.text_area("Descripción")

    age = st.number_input("Edad", 0, 200, 12)
    gender = st.selectbox("Género", [1, 2])

    if st.button("Predecir"):

        if uploaded_file is not None and text != "":

            image = Image.open(uploaded_file)

            # features
            img_feat = extract_image_features(image)
            text_feat = extract_text_features(text)

            tabular = np.array([age, gender])

            # 🔥 junta TODO (ajustaremos esto luego EXACTO)
            features = np.hstack([tabular, text_feat, img_feat]).reshape(1, -1)

            pred = predict_adoption_time(features)

            st.success(f"⏳ Se adoptará en aproximadamente {int(pred)} días")

        else:
            st.error("Falta imagen o descripción")

if __name__ == "__main__":
    main()