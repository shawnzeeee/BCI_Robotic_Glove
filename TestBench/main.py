import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from window_processing import (
    process_idle_windows, process_attention_windows,
    extract_csp_idle_windows, extract_csp_attention_windows
)
from models import (
    get_csp_svm_pipeline, get_csp_lda_pipeline, get_csp_xgb_pipeline,
    get_svm_pipeline, get_xgb_pipeline, get_label_encoder
)
from plotting import plot_confusion_matrix, plot_f1_scores

def main():
    # Load calibration.csv
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Filter', 'Muse', 'calibration.csv')
    df = pd.read_csv(csv_path)

    # Get indices for each class
    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()

    # Feature extraction
    all_output_data = []
    all_output_data.extend(process_attention_windows(attention_indices, df))
    all_output_data.extend(process_idle_windows(idle_indices, df))
    all_output_data = np.array(all_output_data)
    X = all_output_data[:, :-1]
    y = all_output_data[:, -1]

    # CSP windows
    idle_windows, idle_labels = extract_csp_idle_windows(idle_indices, df)
    attention_windows, attention_labels = extract_csp_attention_windows(attention_indices, df)
    X_csp = np.stack(idle_windows + attention_windows)
    y_csp = np.array(idle_labels + attention_labels)
    X_csp_train, X_csp_test, y_csp_train, y_csp_test = train_test_split(X_csp, y_csp, test_size=0.3, random_state=42)

    # --- Models ---
    pipeline_csp_svm = get_csp_svm_pipeline()
    pipeline_csp_svm.fit(X_csp_train, y_csp_train)
    y_pred_csp_svm = pipeline_csp_svm.predict(X_csp_test)
    plot_confusion_matrix(y_csp_test, y_pred_csp_svm, [1, 2], 'CSP + SVM Confusion Matrix (Test)', cmap='Blues')

    pipeline_csp_lda = get_csp_lda_pipeline()
    pipeline_csp_lda.fit(X_csp_train, y_csp_train)
    y_pred_csp_lda = pipeline_csp_lda.predict(X_csp_test)
    plot_confusion_matrix(y_csp_test, y_pred_csp_lda, [1, 2], 'CSP + LDA Confusion Matrix (Test)', cmap='Greens')

    le = get_label_encoder()
    y_csp_train_xgb = le.fit_transform(y_csp_train)
    y_csp_test_xgb = le.transform(y_csp_test)
    pipeline_csp_xgb = get_csp_xgb_pipeline()
    pipeline_csp_xgb.fit(X_csp_train, y_csp_train_xgb)
    y_pred_csp_xgb = pipeline_csp_xgb.predict(X_csp_test)
    plot_confusion_matrix(y_csp_test_xgb, y_pred_csp_xgb, [0, 1], 'CSP + XGBoost Confusion Matrix (Test)', cmap='Purples')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    svm2 = get_svm_pipeline()
    svm2.fit(X_train, y_train)
    y_pred_svm = svm2.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_svm, [1, 2], 'SVM (No CSP) Confusion Matrix (Test)', cmap='Oranges')

    xgb2 = get_xgb_pipeline()
    le2 = get_label_encoder()
    y_train_xgb2 = le2.fit_transform(y_train)
    y_test_xgb2 = le2.transform(y_test)
    xgb2.fit(X_train, y_train_xgb2)
    y_pred_xgb2 = xgb2.predict(X_test)

    # --- F1 Scores ---
    f1_csp_svm = f1_score(y_csp_test, y_pred_csp_svm, average='weighted')
    f1_csp_lda = f1_score(y_csp_test, y_pred_csp_lda, average='weighted')
    f1_csp_xgb = f1_score(y_csp_test_xgb, y_pred_csp_xgb, average='weighted')
    f1_svm2 = f1_score(y_test, y_pred_svm, average='weighted')
    f1_xgb2 = f1_score(y_test_xgb2, y_pred_xgb2, average='weighted')

    model_names = ['CSP+SVM', 'CSP+LDA', 'CSP+XGBoost', 'SVM (No CSP)', 'XGBoost (No CSP)']
    f1_scores = [f1_csp_svm, f1_csp_lda, f1_csp_xgb, f1_svm2, f1_xgb2]
    plot_f1_scores(model_names, f1_scores)

if __name__ == "__main__":
    main()
