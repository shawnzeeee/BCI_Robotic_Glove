import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd

from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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

def process_idle_windows(idle_indices, df, window_size=500, num_windows=4):
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

def extract_csp_idle_windows(idle_indices, df, window_size=500, num_windows=4):
    """
    Extracts raw EEG windows for CSP from idle indices.
    Returns: list of windows (channels x samples), list of labels (all 1)
    """
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    windows = []
    labels = []
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T  # shape: (channels, samples)
            windows.append(window)
            labels.append(1)
    return windows, labels

def extract_csp_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    """
    Extracts raw EEG windows for CSP from attention indices.
    Returns: list of windows (channels x samples), list of labels (all 2)
    """
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    windows = []
    labels = []
    for start_idx in attention_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T  # shape: (channels, samples)
            windows.append(window)
            labels.append(2)
    return windows, labels

# Load your CSV file (replace with your actual CSV path)
csv_path = os.path.join(os.path.dirname(__file__), 'calibration.csv')
df = pd.read_csv(csv_path)

# Get indices where class is 2 (attention) and 1 (idle)
attention_indices = df.index[df['Class'] == 2].tolist()
idle_indices = df.index[df['Class'] == 1].tolist()

all_output_data.extend(process_attention_windows(attention_indices, df))
all_output_data.extend(process_idle_windows(idle_indices, df))

all_output_data = np.array(all_output_data)

X = all_output_data[:, :-1]
y = all_output_data[:, -1]


# Collect raw windows and labels for CSP

window_size = 500  # samples per window
num_channels = 4
channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]

idle_windows, idle_labels = extract_csp_idle_windows(idle_indices, df)
attention_windows, attention_labels = extract_csp_attention_windows(attention_indices, df)

X_csp = np.stack(idle_windows + attention_windows)
y_csp = np.array(idle_labels + attention_labels)

# For CSP (raw windows)
X_csp_train, X_csp_test, y_csp_train, y_csp_test = train_test_split(X_csp, y_csp, test_size=0.3, random_state=42)

print(X_csp.shape)
# --- 1. CSP + SVM ---
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
svm = SVC(kernel='linear')
pipeline_csp_svm = Pipeline([
    ('csp', csp),
    ('svm', svm)
])
pipeline_csp_svm.fit(X_csp_train, y_csp_train)
y_pred_csp_svm = pipeline_csp_svm.predict(X_csp_test)
cm_csp_svm = confusion_matrix(y_csp_test, y_pred_csp_svm)
ConfusionMatrixDisplay(confusion_matrix=cm_csp_svm, display_labels=[1, 2]).plot(cmap=plt.cm.Blues)
plt.title('CSP + SVM Confusion Matrix (Test)')
plt.show()

# --- 2. CSP + LDA ---
lda = LinearDiscriminantAnalysis()
pipeline_csp_lda = Pipeline([
    ('csp', csp),
    ('lda', lda)
])
pipeline_csp_lda.fit(X_csp_train, y_csp_train)
y_pred_csp_lda = pipeline_csp_lda.predict(X_csp_test)
cm_csp_lda = confusion_matrix(y_csp_test, y_pred_csp_lda)
ConfusionMatrixDisplay(confusion_matrix=cm_csp_lda, display_labels=[1, 2]).plot(cmap=plt.cm.Greens)
plt.title('CSP + LDA Confusion Matrix (Test)')
plt.show()

# --- 4. CSP + XGBoost ---
# Remap labels from [1,2] to [0,1] for XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_csp_train_xgb = le.fit_transform(y_csp_train)
y_csp_test_xgb = le.transform(y_csp_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
pipeline_csp_xgb = Pipeline([
    ('csp', csp),
    ('xgb', xgb)
])
pipeline_csp_xgb.fit(X_csp_train, y_csp_train_xgb)
y_pred_csp_xgb = pipeline_csp_xgb.predict(X_csp_test)
# Inverse transform predictions for confusion matrix
cm_csp_xgb = confusion_matrix(y_csp_test_xgb, pipeline_csp_xgb.predict(X_csp_test))
ConfusionMatrixDisplay(confusion_matrix=cm_csp_xgb, display_labels=[0, 1]).plot(cmap=plt.cm.Purples)
plt.title('CSP + XGBoost Confusion Matrix (Test)')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. SVM on features only
svm2 = SVC(kernel='linear')
svm2.fit(X_train, y_train)
y_pred_svm = svm2.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=[1, 2]).plot(cmap=plt.cm.Oranges)
plt.title('SVM (No CSP) Confusion Matrix (Test)')
plt.show()

# XGBoost (No CSP) on features only
xgb2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# Remap labels for XGBoost (No CSP)
le2 = LabelEncoder()
y_train_xgb2 = le2.fit_transform(y_train)
y_test_xgb2 = le2.transform(y_test)
xgb2.fit(X_train, y_train_xgb2)
y_pred_xgb2 = xgb2.predict(X_test)
f1_xgb2 = f1_score(y_test_xgb2, y_pred_xgb2, average='weighted')

# Calculate F1 scores for each model
f1_csp_svm = f1_score(y_csp_test, y_pred_csp_svm, average='weighted')
f1_csp_lda = f1_score(y_csp_test, y_pred_csp_lda, average='weighted')
f1_csp_xgb = f1_score(y_csp_test_xgb, y_pred_csp_xgb, average='weighted')
f1_svm2 = f1_score(y_test, y_pred_svm, average='weighted')
f1_xgb2 = f1_score(y_test_xgb2, y_pred_xgb2, average='weighted')

# Bar plot
model_names = ['CSP+SVM', 'CSP+LDA', 'CSP+XGBoost', 'SVM (No CSP)', 'XGBoost (No CSP)']
f1_scores = [f1_csp_svm, f1_csp_lda, f1_csp_xgb, f1_svm2, f1_xgb2]

plt.figure(figsize=(10, 5))
plt.bar(model_names, f1_scores, color=['blue', 'green', 'purple', 'orange', 'red'])
plt.ylabel('F1 Score (weighted)')
plt.ylim(0, 1)
plt.title('Model F1 Score Comparison')
plt.show()

