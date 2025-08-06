import numpy as np
import pandas as pd
import torch
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


def process_data(sequence: pd.DataFrame, demographics: pd.DataFrame, encoder: LabelEncoder):
    # preprocess
    full_df = pd.merge(
        left=sequence,
        right=demographics,
        on="subject",
        how="left",
    )

    non_target_gestures = full_df[full_df["sequence_type"] == "Non-Target"]["gesture"].unique()
    target_gestures = full_df[full_df["sequence_type"] == "Target"]["gesture"].unique()

    filtered_df = full_df[full_df["phase"] == "Gesture"]
    filtered_df.loc[filtered_df["sequence_type"] == "Non-Target", "gesture"] = non_target_gestures[0]
    agg_recipe = {
        "gesture": ["first"],
        "subject": ["first"],
        "acc_x": ["mean", "std"],
        "acc_y": ["mean", "std"],
        "acc_z": ["mean", "std"],
    }

    filtered_df = filtered_df.groupby("sequence_id")[list(agg_recipe.keys())].agg(agg_recipe)  # type: ignore
    filtered_df.columns = ["_".join(col).strip() if col[1] else col[0] for col in filtered_df.columns.values]
    filtered_df = filtered_df.rename(
        columns={
            "gesture_first": "target",
            "subject_first": "subject",
        }
    )

    # encoder
    target_df = filtered_df["target"]

    target_tensor = torch.tensor(encoder.transform(target_df), dtype=torch.long)
    features_tensor = torch.tensor(filtered_df.drop(columns=["target", "subject"]).to_numpy(), dtype=torch.float32)
    target_gestures_encoded = torch.tensor(encoder.transform(target_gestures))

    return features_tensor, target_tensor


if __name__ == "__main__":
    encoder_path = "./models/label_encoder.pkl"
    raw_path = "./data/raw"

    sequence_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
    demographics_df = pd.read_csv(os.path.join(raw_path, "train_demographics.csv"))

    encoder = load_encoder(encoder_path)
    features_tensor, target_tensor = process_data(sequence_df, demographics_df, encoder)
    
    print(features_tensor.shape)