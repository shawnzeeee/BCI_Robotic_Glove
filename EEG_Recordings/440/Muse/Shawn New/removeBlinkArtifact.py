import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import butter, filtfilt

# --- Config ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "trial_1.csv")
sampling_rate = 256  # Hz for Muse 2
channels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
threshold = 100  # µV for blink detection
window_sec = 0.25  # interpolation window ±0.25 sec

# --- Load EEG Data ---
df = pd.read_csv(csv_path)
time = np.arange(len(df)) / sampling_rate

# --- Bandpass filter ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# --- Detect blink artifacts by threshold ---
def detect_blinks(signal, threshold):
    return np.where(np.abs(signal) > threshold)[0]

# --- Interpolate over blink regions ---
def interpolate_blinks(signal, blink_indices, fs, window_sec):
    cleaned = signal.copy()
    half_window = int(fs * window_sec)
    for idx in blink_indices:
        start = max(0, idx - half_window)
        end = min(len(signal), idx + half_window)
        if end - start > 2:
            cleaned[start:end] = np.linspace(signal[start], signal[end - 1], end - start)
    return cleaned

# --- Clean all channels ---
cleaned_data = {}
filtered_data = {}
for ch in channels:
    raw = df[ch].values
    filtered = bandpass_filter(raw, 0.5, 45, sampling_rate)
    blink_indices = detect_blinks(filtered, threshold)
    cleaned = interpolate_blinks(filtered, blink_indices, sampling_rate, window_sec)
    filtered_data[ch] = filtered
    cleaned_data[ch] = cleaned

# --- Plot 1: Filtered EEG ---
plt.figure(figsize=(12, 8))
for i, ch in enumerate(channels):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, filtered_data[ch], label="Raw (Filtered)", color="blue")
    plt.title(f"{ch} - Raw (Filtered)")
    plt.ylabel("Amplitude (µV)")
    if i == len(channels) - 1:
        plt.xlabel("Time (s)")
    plt.grid(True)
plt.tight_layout()
plt.suptitle("Filtered EEG Channels", fontsize=16, y=1.02)
plt.show(block=False)  # <- non-blocking

# --- Plot 2: Cleaned EEG ---
plt.figure(figsize=(12, 8))
for i, ch in enumerate(channels):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, cleaned_data[ch], label="Cleaned", color="orange")
    plt.title(f"{ch} - Cleaned (Artifact Removed)")
    plt.ylabel("Amplitude (µV)")
    if i == len(channels) - 1:
        plt.xlabel("Time (s)")
    plt.grid(True)
plt.tight_layout()
plt.suptitle("Cleaned EEG Channels", fontsize=16, y=1.02)
plt.show()  # This one blocks until you close both
