import numpy as np
import pandas as pd
import librosa
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def get_fft_power(signal: pd.Series):
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    power = power // 2
    return power


def amplitude_to_db_torch(magnitude: torch.Tensor, ref: str = "max", amin: float = 1e-10, top_db: float = 80.0) -> torch.Tensor:
    # magnitude: (N, H, W) or (B, F, H, W)
    mag = torch.clamp(magnitude, min=amin)
    if ref == "max":
        ref_val = mag.max()
    else:
        ref_val = float(ref)
    log_spec = 20.0 * torch.log10(mag) - 20.0 * torch.log10(ref_val)
    if top_db is not None:
        max_val = log_spec.max()
        log_spec = torch.clamp(log_spec, min=max_val - top_db)
    return log_spec


def create_spectrograms_torch(sequences: torch.Tensor, n_fft: int = 16, hop_length: int = 4, to_db: bool = True) -> torch.Tensor:
    """
    sequences: (batch, channels, seq_len)
    returns: (batch, channels, H, W) spectrograms (no resizing)
    """
    device = sequences.device
    batch, channels, seq_len = sequences.shape

    with torch.no_grad():
        reshaped = sequences.view(batch * channels, seq_len)
        window = torch.hann_window(n_fft, device=device, dtype=sequences.dtype)
        stft_result = torch.stft(
            reshaped, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        )
        magnitude = torch.abs(stft_result)

        if to_db:
            magnitude = amplitude_to_db_torch(magnitude, ref="max")

        _, H, W = magnitude.shape
        specs = magnitude.view(batch, channels, H, W)

    return specs


def resize_spectrograms_torch(specs: torch.Tensor, target_size=(224, 224)) -> torch.Tensor:
    """
    specs: (batch, channels, H, W)
    returns: (batch, channels, target_h, target_w)
    """
    with torch.no_grad():
        resized = F.interpolate(specs, size=target_size, mode="bilinear", align_corners=False)
    return resized

def remove_gravity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates linear acceleration by removing the gravity component from the raw
    accelerometer data using the device's orientation.

    Args:
        df (pd.DataFrame): The time-step level DataFrame which must include
                           'acc_x', 'acc_y', 'acc_z', and the quaternion columns
                           'rot_w', 'rot_x', 'rot_y', 'rot_z'.

    Returns:
        pd.DataFrame: The original DataFrame with new columns for linear
                      acceleration ('linear_acc_x', 'linear_acc_y', 'linear_acc_z',
                      'linear_acc_mag').
    """
    gravity = np.array([0, 0, -9.81])
    quat = df[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy()
    accel = df[["acc_x", "acc_y", "acc_z"]].to_numpy()

    rotations = Rotation.from_quat(quat)
    world_accel = rotations.apply(accel, inverse=True)
    linear_accelerations_world = world_accel - gravity

    linear_accelerations_device = rotations.apply(linear_accelerations_world)

    # 5. Add the new features back to the DataFrame
    df["acc_x"] = linear_accelerations_device[:, 0]
    df["acc_y"] = linear_accelerations_device[:, 1]
    df["acc_z"] = linear_accelerations_device[:, 2]

    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dummy_input = torch.rand(7000, 17, 35, dtype=torch.float32)
    # Step 1: create spectrograms
    spectrogram_batch = create_spectrograms_torch(dummy_input)
    print("Raw spectrograms:", spectrogram_batch.shape)

    # Step 2: resize (could be done in smaller chunks inside training loop)
    resized_batch = resize_spectrograms_torch(spectrogram_batch[:64, :, :], target_size=(224, 224))
    print("Resized spectrograms:", resized_batch.shape)
