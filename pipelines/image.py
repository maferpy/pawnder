# pipelines/image.py
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -------------------------
# CARGA DEL MODELO (CACHEADO)
# -------------------------
@st.cache_resource
def load_image_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

image_model = load_image_model()

import joblib

@st.cache_resource
def load_pca():
    return joblib.load("models/pca_200.pkl")

pca = load_pca()

# -------------------------
# EXTRACCIÓN DE FEATURES
# -------------------------
def get_image_features(image: Image.Image) -> np.ndarray:
    """
    Convierte una imagen PIL en un vector de características usando EfficientNetB0.
    """
    # Redimensionar la imagen al tamaño esperado por EfficientNet
    image = image.resize((224, 224))
    
    # Convertir a array y expandir dimensiones
    img_array = np.array(image)
    if img_array.shape[-1] == 4:  # eliminar canal alpha si existe
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocesar
    img_array = preprocess_input(img_array)
    
    # Extraer features sin mostrar barra de progreso
    features = image_model.predict(img_array, verbose=0)
    
    return features

# -------------------------
# PIPELINE COMPLETO
# -------------------------
def process_image_pipeline(uploaded_file):
    # Abrir imagen
    image = Image.open(uploaded_file).convert("RGB")
    
    # Extraer features originales
    features = get_image_features(image)  # shape (1280,)
    
    # Aplicar PCA
    features_pca = pca.transform(features.reshape(1, -1))  # reshape necesario para 2D
    
    return features_pca  # shape (1, 200)

