import numpy as np
from scipy.signal import welch

def calculate_hjorth_parameters(signal):
    # ...existing code from performance.py...
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    variance = np.var(signal)
    mobility = np.sqrt(np.var(first_derivative) / variance)
    complexity = np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / mobility
    return mobility, complexity

def calculate_bandpowers(signal, fs=250):
    # ...existing code from performance.py...
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
    beta_band = np.logical_and(freqs >= 13, freqs <= 30)
    alpha_power = np.sum(psd[alpha_band])
    beta_power = np.sum(psd[beta_band])
    return alpha_power, beta_power
