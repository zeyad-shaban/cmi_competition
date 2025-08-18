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
    power = power / 2
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

def remove_gravity(df: pd.DataFrame,
                   quat_cols=("rot_x","rot_y","rot_z","rot_w"),
                   acc_cols=("acc_x","acc_y","acc_z"),
                   gravity_mag=9.81):
    """Return a copy with added linear_acc_x/y/z and linear_acc_mag.
       Writes only to rows with valid quaternion; invalid rows keep original accel.
       Expects quat order (x,y,z,w).
    """
    df = df.copy()

    quat_df = df[list(quat_cols)]
    # valid if no NaNs and not all zeros (tolerance for floating)
    valid_mask = (~quat_df.isnull().any(axis=1)) & (~np.isclose(quat_df.values, 0).all(axis=1))
    valid_idx = df.index[valid_mask]

    # By default keep original accel values for invalid rows
    df["linear_acc_x"] = df[acc_cols[0]].astype(float)
    df["linear_acc_y"] = df[acc_cols[1]].astype(float)
    df["linear_acc_z"] = df[acc_cols[2]].astype(float)

    if len(valid_idx) > 0:
        quat_vals = quat_df.loc[valid_idx].to_numpy(dtype=float)     # shape M x 4 (x,y,z,w)
        accel_vals = df.loc[valid_idx, list(acc_cols)].to_numpy(dtype=float)  # shape M x 3

        rotations = Rotation.from_quat(quat_vals)                            # expects [x,y,z,w]
        accel_world = rotations.apply(accel_vals, inverse=True)       # sensor -> world

        gravity = np.array([0.0, 0.0, gravity_mag], dtype=float)
        linear_world = accel_world - gravity                          # linear accel in world frame

        # write results back only to valid rows
        df.loc[valid_idx, "linear_acc_x"] = linear_world[:, 0]
        df.loc[valid_idx, "linear_acc_y"] = linear_world[:, 1]
        df.loc[valid_idx, "linear_acc_z"] = linear_world[:, 2]

    df["linear_acc_mag"] = np.linalg.norm(df[["linear_acc_x","linear_acc_y","linear_acc_z"]].to_numpy(), axis=1)
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
