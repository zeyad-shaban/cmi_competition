import numpy as np
import pandas as pd

def get_fft_power(signal: pd.Series):
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    power = power // 2

    return power