import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder


def load_encoder(encoder_path, df: pd.DataFrame | None = None):
    if df is None:
        assert os.path.exists(encoder_path), "no prefitted encoder found, please provide DF to create new encoder"
        return joblib.load(encoder_path)

    encoder = LabelEncoder()
    encoder.fit(df)
    joblib.dump(encoder, encoder_path)
    return encoder


def get_preprocessed(split="train"):
    full_df = pd.merge(
        left=pd.read_csv(f"./data/raw/{split}.csv"),
        right=pd.read_csv(f"./data/raw/{split}_demographics.csv"),
        on="subject",
        how="left",
    )

    return full_df


if __name__ == "__main__":
    splits = ["train", "test"]
    for split in splits:
        preprocessed_df = get_preprocessed(split)
        preprocessed_df.to_csv(f"./data/processed/{split}.csv")
