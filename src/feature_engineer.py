import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder


def load_encoder(encoder_path, df: pd.DataFrame | None = None):
    if os.path.exists(encoder_path):
        print("prefitted encoder found...")
        return joblib.load(encoder_path)

    print("prefitted encoder not found, creating new")
    assert df is not None, "DataFrame expected as encoder_path doesn't exist"
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
