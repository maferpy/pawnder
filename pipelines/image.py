import numpy as np
import joblib
import ssl
import certifi  
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Modelo CNN EfficientNetB0
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

image_model = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output)
)

pca_200 = joblib.load("models/pca_200.pkl")

def preprocess_image(image):

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image

def get_image_embedding(image):

    img = preprocess_image(image)

    features = image_model.predict(img, verbose=0)

    return features

def get_image_features(image):

    emb = get_image_embedding(image)

    emb_reduced = pca_200.transform(emb)

    return emb_reduced

def process_image_pipeline(image):

    features = get_image_features(image)

    img_df = pd.DataFrame(features)
    img_df.columns = [f"img_{i}" for i in range(img_df.shape[1])]

    return img_df