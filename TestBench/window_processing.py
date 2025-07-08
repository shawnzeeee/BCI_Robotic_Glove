import numpy as np
from feature_extraction import calculate_hjorth_parameters, calculate_bandpowers

def process_idle_windows(idle_indices, df, window_size=500, num_windows=4):
    # ...existing code from performance.py...
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
            features.append(1)
            processed_data.append(features)
    return processed_data

def process_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    # ...existing code from performance.py...
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
            features.append(2)
            processed_data.append(features)
    return processed_data

def extract_csp_idle_windows(idle_indices, df, window_size=500, num_windows=4):
    # ...existing code from performance.py...
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    windows = []
    labels = []
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T
            windows.append(window)
            labels.append(1)
    return windows, labels

def extract_csp_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    # ...existing code from performance.py...
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    windows = []
    labels = []
    for start_idx in attention_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T
            windows.append(window)
            labels.append(2)
    return windows, labels
