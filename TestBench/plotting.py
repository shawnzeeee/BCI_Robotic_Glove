import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# def plot_confusion_matrix(y_true, y_pred, display_labels, title, cmap='Blues'):
#     cm = confusion_matrix(y_true, y_pred)
#     ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(cmap=getattr(plt.cm, cmap))
#     plt.title(title)
#     plt.show()

def plot_f1_scores(model_names, f1_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, f1_scores, color=['blue', 'green', 'purple', 'orange', 'red'])
    plt.ylabel('F1 Score (weighted)')
    plt.ylim(0, 1)
    plt.title('Model F1 Score Comparison')
    plt.show()
