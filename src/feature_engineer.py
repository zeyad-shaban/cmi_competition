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


# Feature Engineer
def rotation_feature_engineer(df: pd.DataFrame):
    df = df.copy()
    quat_arr = df[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    num_samples = quat_arr.shape[0]
    angular_vel = np.zeros([num_samples, 3])

    rotation_object = Rotation.from_quat(quat_arr)
    rotation_vectors = rotation_object.as_rotvec()

    df["rotvec_x"] = rotation_vectors[:, 0]
    df["rotvec_y"] = rotation_vectors[:, 1]
    df["rotvec_z"] = rotation_vectors[:, 2]
    df["rot_angle"] = np.linalg.norm(rotation_vectors, axis=1)

    dt = 1 / 200
    num_samples = len(df)

    if num_samples > 1:
        q_current = quat_arr[:-1]  # t0 to t_{n-2}
        q_next = quat_arr[1:]  # t1 to t_{n-1}

        rot_curr = Rotation.from_quat(q_current)
        rot_next = Rotation.from_quat(q_next)
        delta_rot = rot_curr.inv() * rot_next

        angular_vel[1:] = delta_rot.as_rotvec() / dt

    df["angular_vel_x"] = angular_vel[:, 0]
    df["angular_vel_y"] = angular_vel[:, 1]
    df["angular_vel_z"] = angular_vel[:, 2]
    df["angular_speed"] = np.linalg.norm(angular_vel, axis=1)

    return df


def accelrometer_feature_engineer(df: pd.DataFrame):
    cols_of_interest = ["linear_acc_x", "linear_acc_y", "linear_acc_z"]

    for col in cols_of_interest:
        df[f"fft_{col}"] = df.groupby("sequence_id")[col].transform(get_fft_power)

    df["acc_mag"] = np.linalg.norm([df["acc_x"], df["acc_y"], df["acc_z"]], axis=0)
    df["linear_acc_mag"] = np.linalg.norm([df["linear_acc_x"], df["linear_acc_y"], df["linear_acc_z"]], axis=0)
    df["jerk_acc_x"] = df.groupby("sequence_id")["linear_acc_x"].diff().fillna(0)
    df["jerk_acc_y"] = df.groupby("sequence_id")["linear_acc_y"].diff().fillna(0)
    df["jerk_acc_z"] = df.groupby("sequence_id")["linear_acc_z"].diff().fillna(0)
    df["jerk_mag"] = np.linalg.norm([df["jerk_acc_x"], df["jerk_acc_y"], df["jerk_acc_z"]], axis=0)
    return df


if __name__ == "__main__":
    encoder_path = "./models/label_encoder.pkl"
    raw_path = "./data/raw"

    sequence_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
    demographics_df = pd.read_csv(os.path.join(raw_path, "train_demographics.csv"))

    encoder = load_encoder(encoder_path)
    features_tensor, target_tensor = process_data(sequence_df, demographics_df, encoder)

    print(features_tensor.shape)
