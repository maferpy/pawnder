import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer



# Modelo de embeddings
MODEL_NAME = "intfloat/multilingual-e5-base"
model = SentenceTransformer(MODEL_NAME)

# Limpieza de texto
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Features basicas
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


# Keywords
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


# Embeddings

def get_embeddings(df):
    texts = df["Description"].tolist()

    # IMPORTANTE: inference mode (sin training)
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embeddings


# PIPELINE COMPLETO

def process_text_pipeline(df):

    df = df.copy()
    for col in ["age_bin", "fee_pets"]:
        df[col] = df[col].astype("category")

    # 1. asegurar columna
 
    df["Description"] = df["Description"].fillna("")


    # 2. limpieza

    df["Description"] = df["Description"].apply(clean_text)


    # 3. features manuales

    df = basic_features(df)
    df = keyword_features(df)


    # 4. embeddings

    embeddings = get_embeddings(df)

    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]


    # 5. merge final
 
    df_final = pd.concat(
        [df.reset_index(drop=True), emb_df],
        axis=1
    )

    return df_final