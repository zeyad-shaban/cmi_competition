import numpy as np
import pandas as pd
import torch
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from src.utils.math_utils import get_fft_power
from scipy.spatial.transform import Rotation


def load_encoder(encoder_path, df: pd.DataFrame | None = None):
    if df is None:
        assert os.path.exists(encoder_path), "no prefitted encoder found, please provide DF to create new encoder"
        return joblib.load(encoder_path)

    encoder = LabelEncoder()
    encoder.fit(df)
    joblib.dump(encoder, encoder_path)
    return encoder


def rotation_feature_engineer(df: pd.DataFrame):
    quat = df[["rot_w", "rot_x", "rot_y", "rot_z"]]
    rotation_object = Rotation.from_quat(quat)
    rotation_vectors = rotation_object.as_rotvec()

    df["rotvec_x"] = rotation_vectors[:, 0]
    df["rotvec_y"] = rotation_vectors[:, 1]
    df["rotvec_z"] = rotation_vectors[:, 2]
    df["rot_angle"] = np.linalg.norm(rotation_vectors, axis=1)

    df["rotvec_y_diff"] = df.groupby("sequence_id")["rotvec_y"].transform(lambda x: x.diff().fillna(0))
    df["rotvec_x_diff"] = df.groupby("sequence_id")["rotvec_x"].transform(lambda x: x.diff().fillna(0))
    df["rotvec_z_diff"] = df.groupby("sequence_id")["rotvec_z"].transform(lambda x: x.diff().fillna(0))
    df["angular_mag"] = np.linalg.norm([df["rotvec_x_diff"], df["rotvec_y_diff"], df["rotvec_z_diff"]], axis=0)
    
    return df

def accelrometer_feature_engineer(df: pd.DataFrame):
    cols_of_interest = ["acc_x", "acc_y", "acc_z"]

    for col in cols_of_interest:
        df[f"fft_{col}"] = df.groupby("sequence_id")[col].transform(get_fft_power)

    df["acc_mag"] = np.linalg.norm([df["acc_x"], df["acc_y"], df["acc_z"]], axis=0)
    df["jerk_acc_x"] = df.groupby("sequence_id")["acc_x"].diff().fillna(0)
    df["jerk_acc_y"] = df.groupby("sequence_id")["acc_y"].diff().fillna(0)
    df["jerk_acc_z"] = df.groupby("sequence_id")["acc_z"].diff().fillna(0)
    
    return df


if __name__ == "__main__":
    encoder_path = "./models/label_encoder.pkl"
    raw_path = "./data/raw"

    sequence_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
    demographics_df = pd.read_csv(os.path.join(raw_path, "train_demographics.csv"))

    encoder = load_encoder(encoder_path)
    features_tensor, target_tensor = process_data(sequence_df, demographics_df, encoder)
    
    print(features_tensor.shape)