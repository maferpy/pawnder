import numpy as np
import joblib

model = joblib.load("models/stacking_model.pkl")
pca = joblib.load("models/pca_200.pkl")

fee_cfg = joblib.load("models/fee_bins_u.pkl")
fee_bins = fee_cfg["bins"]
fee_labels = fee_cfg["labels"]

model_text = model["model_text"]
model_img = model["model_img"]
meta_model = model["meta_model"]

text_tab_cols = model["text_cols"]
img_cols = model["img_cols"]

def predict_adoption_time(text_features, img_features):
    
    # 1. predicciones base
    pred_text = model_text.predict_proba(text_features)
    pred_img = model_img.predict_proba(img_features)

    # 2. stacking
    X_meta = np.hstack([pred_text, pred_img])

    # 3. predicción final
    pred = meta_model.predict(X_meta)

    return pred[0]
    