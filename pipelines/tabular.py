import pandas as pd
import numpy as np
import joblib

# cargar bins y labels de fee   
fee_cfg = joblib.load("models/fee_bins_u.pkl")
fee_bins = fee_cfg["bins"]
fee_labels = fee_cfg["labels"]

# cargar diccionario de breeds
breed_dict = pd.read_csv("breeds/BreedLabels.csv")
breed_map = dict(zip(breed_dict["BreedName"], breed_dict["BreedID"]))


# Mapeo de breeds
def map_breed(df):

    df["Breed1"] = df["Breed1"].str.lower().fillna("unknown")

    df["Breed1"] = df["Breed1"].apply(
        lambda x: breed_map.get(x, 307)  # 307 = unknown
    )

    return df


# Feature engineering
def df_engineering(df, fee_bins, fee_labels):

    # colors
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
    df["age_bin"] = df["age_bin"].astype("category")
    
    # fee_pets

    df["fee_pets"] = pd.cut(
        df["Fee"],
        bins=fee_bins,
        labels=fee_labels,
        include_lowest=True
    )
    df["fee_pets"] = df["fee_pets"].astype("category")
    # is_pure
    df["is_pure"] = (df["Breed1"] != 307).astype(int)


    # health
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


    # log transform
    df["log_age"] = np.log1p(df["Age"])

    return df


# Pipeline tabular
def process_tabular_pipeline(df):

    df = df.copy()

    # limpieza básica
    df["Breed1"] = df["Breed1"].fillna("unknown")

    # mapping
    df = map_breed(df)

    # feature engineering
    df = df_engineering(df, fee_bins, fee_labels)

    return df
