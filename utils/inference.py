import numpy as np
import joblib

model = joblib.load("models/model.pkl")
pca = joblib.load("models/pca_200.pkl")

def predict_adoption_time(features):
    pred = model.predict(features)
    return pred[0]