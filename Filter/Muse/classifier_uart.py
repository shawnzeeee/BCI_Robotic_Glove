import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd
from sklearn.svm import SVC
import joblib

# Function to calculate mobility and complexity (Hjorth parameters)
def calculate_hjorth_parameters(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    variance = np.var(signal)
    mobility = np.sqrt(np.var(first_derivative) / variance)
    complexity = np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / mobility
    return mobility, complexity

# Function to calculate bandpowers (alpha and beta)
def calculate_bandpowers(signal, fs=250):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
    beta_band = np.logical_and(freqs >= 13, freqs <= 30)
    alpha_power = np.sum(psd[alpha_band])
    beta_power = np.sum(psd[beta_band])
    return alpha_power, beta_power

all_output_data = []

def process_idle_windows(idle_indices, df, window_size=500, num_windows=5):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end]
            features = []
            for channel in channel_names:
                signal = window[channel].values
                mobility, complexity = calculate_hjorth_parameters(signal)
                alpha_power, beta_power = calculate_bandpowers(signal)
                features.extend([mobility, complexity, alpha_power, beta_power])
            actual_class = df.iloc[window_start, 4]
            features.append(1)
            processed_data.append(features)
    return processed_data

def process_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in attention_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end]
            features = []
            for channel in channel_names:
                signal = window[channel].values
                mobility, complexity = calculate_hjorth_parameters(signal)
                alpha_power, beta_power = calculate_bandpowers(signal)
                features.extend([mobility, complexity, alpha_power, beta_power])
            actual_class = df.iloc[window_start, 4]
            features.append(2)
            processed_data.append(features)
    return processed_data

# Load your CSV file (replace with your actual CSV path)
csv_path = os.path.join(os.path.dirname(__file__), 'calibration.csv')
df = pd.read_csv(csv_path)

# Get indices where class is 2 (attention) and 1 (idle)
attention_indices = df.index[df['Class'] == 2].tolist()
idle_indices = df.index[df['Class'] == 1].tolist()

all_output_data.extend(process_attention_windows(attention_indices, df))
all_output_data.extend(process_idle_windows(idle_indices, df))

all_output_data = np.array(all_output_data)


# Features: all columns except last
X = all_output_data[:, :-1]
# Labels: last column
y = all_output_data[:, -1]

svm = SVC(kernel='linear')

# Train SVM directly on features
svm.fit(X, y)

print("SVM training complete.")

# Export the trained SVM model
model_path = os.path.join(os.path.dirname(__file__), 'svm_model.joblib')
joblib.dump(svm, model_path)
print(f"SVM model exported to {model_path}")

file_path = os.path.join(os.path.dirname(__file__), 'data.bin')

print('[INFO] Reading EEG data from data.bin...')

window_size = 2000  # 2 seconds, 4 channels, 250Hz

try:
    attention_threshold = 0
    while True:
        data_array = np.fromfile(file_path, dtype=np.float32)
        if len(data_array) < window_size:
            time.sleep(0.1)
            continue
        if len(data_array) == window_size:
            eeg_window = data_array.reshape(500, 4)
            features = []
            for ch in range(4):
                signal = eeg_window[:, ch]
                #print(signal)
                mobility, complexity = calculate_hjorth_parameters(signal)
                alpha_power, beta_power = calculate_bandpowers(signal)
                features.extend([mobility, complexity, alpha_power, beta_power])
            features = np.array(features).reshape(1, -1)
            # Predict class
            predicted_class = svm.predict(features)[0]
            adder = -10
            if predicted_class == 2:
                adder = 1

            attention_threshold += adder
            attention_threshold = max(0, min(attention_threshold, 300))

            gesture = "Open"
            if attention_threshold >= 200:
                gesture = "Close"
            print(f"Predicted class: {gesture}")
        #time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting...")