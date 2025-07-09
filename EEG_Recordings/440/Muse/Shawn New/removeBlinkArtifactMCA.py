import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import cvxpy as cp
import os

# --- Config ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "trial_1.csv")
fs = 256  # Hz
channels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
lam = 0.1  # regularization strength

# --- Load EEG ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / fs

# --- STFT Basis Generator (complex) ---
def generate_stft_basis(signal, fs, win_sec, hop_sec):
    nperseg = int(fs * win_sec)
    noverlap = nperseg - int(fs * hop_sec)
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return Zxx.flatten(), Zxx.shape, nperseg, noverlap

# --- MCA Solver (real + imag parts) ---
def solve_mca_complex(y_vec, shape, lam):
    n = y_vec.shape[0]
    alpha1 = cp.Variable(n, complex=True)
    alpha2 = cp.Variable(n, complex=True)
    objective = cp.Minimize(lam * cp.norm1(alpha1) + lam * cp.norm1(alpha2))
    constraints = [cp.real(alpha1 + alpha2) == np.real(y_vec),
                   cp.imag(alpha1 + alpha2) == np.imag(y_vec)]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return alpha1.value.reshape(shape), alpha2.value.reshape(shape)

# --- ISTFT Reconstructor ---
def reconstruct_from_stft(stft_matrix, fs, nperseg, noverlap):
    _, x = istft(stft_matrix, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return x

# --- Process All Channels ---
cleaned_signals = {}
for ch in channels:
    print(f"Processing {ch}...")
    raw = df[ch].values

    # Build STFT for EEG and blink-like windows
    y_eeg, shape, nseg, novl = generate_stft_basis(raw, fs, win_sec=2.0, hop_sec=1.5)
    # y_blink could be generated for adaptive λ or shape alignment if needed

    # Solve MCA and keep EEG component
    alpha1, _ = solve_mca_complex(y_eeg, shape, lam=lam)

    # Reconstruct cleaned signal
    cleaned = reconstruct_from_stft(alpha1, fs, nperseg=nseg, noverlap=novl)
    cleaned = cleaned[:len(raw)]  # Trim
    cleaned_signals[ch] = cleaned

# --- Plot 1: Raw EEG ---
plt.figure(figsize=(12, 10))
for i, ch in enumerate(channels):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, df[ch].values, color="blue", label="Raw")
    plt.title(f"{ch} – Raw EEG")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    if i == len(channels) - 1:
        plt.xlabel("Time (s)")
    plt.legend()
plt.tight_layout()
plt.suptitle("Raw EEG - All Channels", fontsize=16, y=1.02)
plt.show(block=False)

# --- Plot 2: Cleaned EEG ---
plt.figure(figsize=(12, 10))
for i, ch in enumerate(channels):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, cleaned_signals[ch], color="orange", label="Cleaned")
    plt.title(f"{ch} – Cleaned EEG (MCA)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    if i == len(channels) - 1:
        plt.xlabel("Time (s)")
    plt.legend()
plt.tight_layout()
plt.suptitle("Cleaned EEG (MCA) - All Channels", fontsize=16, y=1.02)
plt.show(block=False)

input("Press Enter to close the plots...")
