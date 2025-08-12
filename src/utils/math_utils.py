import numpy as np
import pandas as pd
import librosa
import torch
import torchvision.transforms as transforms


def get_fft_power(signal: pd.Series):
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    power = power // 2

    return power


def create_and_resize_spectrograms_torch(sequences_tensor: torch.Tensor,
                                         n_fft: int = 16, hop_length: int = 4,
                                         target_size=(224, 224)) -> torch.Tensor:
    B, F, T = sequences_tensor.shape
    reshaped_tensor = sequences_tensor.view(B * F, T)
    window = torch.hann_window(n_fft).to(sequences_tensor.device)

    stft_result = torch.stft(reshaped_tensor, n_fft=n_fft, hop_length=hop_length,
                              window=window, return_complex=True)
    magnitude = torch.abs(stft_result)
    _, H, W = magnitude.shape
    specs = magnitude.view(B, F, H, W)

    resizer = transforms.Resize(target_size, antialias=True)
    resized_list = [resizer(item) for item in specs]  # each item: (F, H, W)
    resized_batch = torch.stack(resized_list, dim=0)
    return resized_batch

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dummy_input = torch.rand(64, 17, 35)
    spectrogram_batch = create_and_resize_spectrograms_torch(dummy_input)
    print(spectrogram_batch.shape)