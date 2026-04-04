import pandas as pd
import numpy as np
import joblib

# cargar diccionario de breeds
breed_dict = pd.read_csv("data/breeds.csv")

breed_map = dict(zip(breed_dict["BreedName"], breed_dict["BreedID"]))


# --------------------------
# 1. MAPEO BREED
# --------------------------
def map_breed(df):

    df["Breed1"] = df["Breed1"].str.lower().fillna("unknown")

    df["Breed1"] = df["Breed1"].apply(
        lambda x: breed_map.get(x, 0)  # 0 = unknown
    )

    return df


# --------------------------
# 2. FEATURE ENGINEERING
# --------------------------
def df_engineering(df):

    # colores activos
    df["num_colors"] = (df[["Color1", "Color2", "Color3"]] > 0).sum(axis=1)

    # age bin
    df["age_bin"] = pd.cut(
        df["Age"],
        bins=[-1, 2, 6, 12, 36, 84, 1000],
        labels=[
            "newborn",
            "puppy",
            "junior",
            "young_adult",
            "adult",
            "senior"
        ]
    )

    # fee bins (dinámico ya precomputado en training idealmente)
    mask = df["Fee"] > 0

    if mask.sum() > 0:
        q50 = df.loc[mask, "Fee"].quantile(0.5)
        q85 = df.loc[mask, "Fee"].quantile(0.85)
    else:
        q50, q85 = 1, 10

    bins = [-1, 0, q50, q85, df["Fee"].max()]
    labels = ["gratis", "bajo", "medio", "alto"]

    df["fee_pets"] = pd.cut(df["Fee"], bins=bins, labels=labels)

    # is_pure
    df["is_pure"] = ((df["Breed2"] == 0) & (df["Breed1"] != 307)).astype(int)

    # health score
    df["health_score"] = (
        df["Vaccinated"] +
        df["Dewormed"] +
        df["Sterilized"]
    )

    df["health_issues"] = (df["Health"] == 3).astype(int)

    df["risk_score"] = (
        df["health_issues"] +
        (df["Age"] > 84).astype(int)
    )

    return df


# --------------------------
# 3. PIPELINE COMPLETO
# --------------------------
def process_tabular_pipeline(df):

    df = df.copy()

    # limpieza básica
    df["Breed1"] = df["Breed1"].fillna("unknown")

    # mapping
    df = map_breed(df)

    # feature engineering
    df = df_engineering(df)

    return df