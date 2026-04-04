import numpy as np
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Modelo CNN 
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = GlobalAveragePooling2D()(base_model.output)
image_model = Model(inputs=base_model.input, outputs=x)

# PCA entrenado
pca = joblib.load("models/pca_200.pkl")


def extract_image_features(image: Image.Image):

    img = image.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = image_model.predict(img, verbose=0)

   
    features = pca.transform(features)

    return features.flatten()