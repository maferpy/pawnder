import numpy as np
import pandas as pd
import re
import streamlit as st
from sentence_transformers import SentenceTransformer

MEAN_EMB = np.load("models/mean_embedding.npy")

# -------------------------
# LIMPIEZA
# -------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------------------------
# FEATURES BÁSICAS
# -------------------------
def basic_features(df):
    df["desc_words"] = df["Description"].apply(lambda x: len(x.split()))
    df["desc_chars"] = df["Description"].apply(len)

    df["is_short"] = (df["desc_words"] < 30).astype(int)
    df["is_long"] = (df["desc_words"] > 80).astype(int)

    df["num_exclamation"] = df["Description"].apply(lambda x: x.count("!"))
    df["has_numbers"] = df["Description"].str.contains(r"\d", na=False).astype(int)
    df["has_uppercase"] = df["Description"].str.contains(r"[A-Z]", na=False).astype(int)

    df["num_sentences"] = df["Description"].str.count(r"[.!?]")

    df["avg_word_length"] = df["Description"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )

    return df


# -------------------------
# KEYWORDS
# -------------------------
def keyword_features(df):

    df["is_friendly"] = df["Description"].str.contains(
        r"friendly|amigable|cariñoso|playful|juguet",
        regex=True,
        na=False
    ).astype(int)

    df["good_with_kids"] = df["Description"].str.contains(
        r"kids?|children|niñ",
        regex=True,
        na=False
    ).astype(int)

    df["good_with_pets"] = df["Description"].str.contains(
        r"dogs?|cats?|perros?|gatos?",
        regex=True,
        na=False
    ).astype(int)

    df["urgent"] = df["Description"].str.contains(
        r"urgent|asap|rescue|urgente",
        regex=True,
        na=False
    ).astype(int)

    df["has_health_issue"] = df["Description"].str.contains(
        r"sick|injured|disease|enfermo|herido",
        regex=True,
        na=False
    ).astype(int)

    return df


# -------------------------
# PIPELINE FINAL (SIN EMBEDDINGS)
# -------------------------

def process_text_pipeline(df):

    df = df.copy()

    # -------------------
    # LIMPIEZA
    # -------------------
    df["Description"] = df["Description"].fillna("").apply(clean_text)

    # -------------------
    # FEATURES BUENAS (rápidas)
    # -------------------
    df = basic_features(df)
    df = keyword_features(df)

    # -------------------
    # 🔥 EMBEDDINGS (NO reales)
    # -------------------
    embeddings = np.tile(MEAN_EMB, (len(df), 1))

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(768)]
    )

    # -------------------
    # CONCAT
    # -------------------
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    return df.drop(columns=["Description"])