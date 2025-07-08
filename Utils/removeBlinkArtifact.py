import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import cvxpy as cp

# --- Parameters ---
fs = 256  # Muse 2 sampling rate
win_len_eeg = 2.0    # Window size for EEG dictionary (sec)
win_len_blink = 0.5  # Window size for blink dictionary (sec)
hop_len = 0.1        # Hop size for STFT (sec)
segment_len = 1024   # Frame length for MCA

# --- STFT Dictionary Generator ---
def create_stft_dictionary(signal, fs, win_len_sec, hop_len_sec):
    nperseg = int(fs * win_len_sec)
    noverlap = int(fs * hop_len_sec)
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    return np.abs(Zxx.reshape(Zxx.shape[0]*Zxx.shape[1], -1)), Zxx.shape

# --- MCA Decomposition using L1 Norm Minimization ---
def mca_decompose(y, D1, D2, lam1=0.1, lam2=0.1):
    alpha1 = cp.Variable(D1.shape[1])
    alpha2 = cp.Variable(D2.shape[1])
    objective = cp.Minimize(lam1 * cp.norm1(alpha1) + lam2 * cp.norm1(alpha2))
    constraints = [D1 @ alpha1 + D2 @ alpha2 == y]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    return alpha1.value, alpha2.value

# --- Clean EEG reconstruction ---
def reconstruct(D, alpha):
    return D @ alpha

# --- Simulated Test Data (Replace this with your Muse data) ---
np.random.seed(0)
t = np.arange(segment_len) / fs
eeg = np.sin(2 * np.pi * 10 * t)          # Simulated EEG (10 Hz)
blink = np.exp(-((t - 1.0)**2) / 0.01)    # Simulated blink at t = 1s
raw = eeg + 2 * blink + 0.1 * np.random.randn(segment_len)

# --- Dictionaries ---
D1, _ = create_stft_dictionary(raw, fs, win_len_eeg, hop_len)
D2, _ = create_stft_dictionary(raw, fs, win_len_blink, hop_len)

# --- MCA ---
alpha1, alpha2 = mca_decompose(raw, D1, D2)

# --- Reconstruction ---
clean = reconstruct(D1, alpha1)

# --- Plotting ---
plt.figure(figsize=(12, 4))
plt.plot(t, raw, label="Raw EEG", alpha=0.6)
plt.plot(t, clean, label="Cleaned EEG", alpha=0.9)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Eye Blink Artifact Removal using MCA")
plt.legend()
plt.tight_layout()
plt.show()
