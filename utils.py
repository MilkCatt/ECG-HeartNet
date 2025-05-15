# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import multilabel_confusion_matrix, f1_score
import seaborn as sns

def count_parameters(model):
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name}: {params} parameters")

def find_optimal_thresholds(y_true, y_probs, thresholds=np.arange(0.05, 0.95, 0.05)):
    """
    Finds optimal threshold for each class based on F1 score.
    """
    n_classes = y_true.shape[1]
    best_thresholds = []
    for i in range(n_classes):
        best_thresh = 0.5
        best_f1 = 0
        for t in thresholds:
            preds = (y_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)

def evaluate_with_custom_thresholds(trainer, dataloader, thresholds):
    trainer.model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    preds = (all_probs >= thresholds).astype(int)

    f1_per_class = f1_score(all_labels, preds, average=None)
    print("F1 score with tuned thresholds:", f1_per_class)

def plot_multilabel_conf_matrices(y_true, y_pred, class_names=None):
    confs = multilabel_confusion_matrix(y_true, y_pred)
    
    for i, cm in enumerate(confs):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        title = f"Class {i}" if class_names is None else f"{class_names[i]}"
        plt.title(f"Confusion Matrix: {title}")
        plt.show()
