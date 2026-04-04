import numpy as np
import re
from sentence_transformers import SentenceTransformer

# mismo modelo que Colab
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Clean text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Features básicas
def basic_features(text):

    words = text.split()

    desc_words = len(words)
    desc_chars = len(text)

    is_short = int(desc_words < 30)
    is_long = int(desc_words > 80)

    num_exclamation = text.count("!")
    has_numbers = int(bool(re.search(r"\d", text)))
    has_uppercase = int(any(c.isupper() for c in text))
    num_sentences = text.count(".") + text.count("!") + text.count("?")

    avg_word_length = (
        np.mean([len(w) for w in words]) if words else 0
    )

    return np.array([
        desc_words,
        desc_chars,
        is_short,
        is_long,
        num_exclamation,
        has_numbers,
        has_uppercase,
        num_sentences,
        avg_word_length
    ])


# Keywords
def keyword_features(text):

    return np.array([
        int(bool(re.search(r"friendly|amigable|cariños|playful|juguet", text))),
        int(bool(re.search(r"kids?|children|niñ", text))),
        int(bool(re.search(r"dogs?|cats?|perros?|gatos?", text))),
        int(bool(re.search(r"urgent|asap|rescue|urgente", text))),
        int(bool(re.search(r"sick|injured|disease|enfermo|herido", text))),
    ])


# Embeddings
def text_embedding(text):
    text = clean_text(text)
    return model.encode([text])[0]


# Pipeline de texto
def extract_text_features(text):

    text = clean_text(text)

    basic = basic_features(text)
    keywords = keyword_features(text)
    emb = text_embedding(text)

    return np.concatenate([basic, keywords, emb])