import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any


def plot_cm(cm: np.ndarray, classes: list[str]) -> None:
    plt.figure(figsize=(13, 13))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=80)
    plt.title("Confusion matrix")


def get_avg_fold(folds_score: list[dict[str, Any]]):
    f1_binary_avg = np.mean([fold["f1_binary"] for fold in folds_score])
    f1_macro_avg = np.mean([fold["f1_macro"] for fold in folds_score])
    competition_avg = np.mean([fold["competition_evaluation"] for fold in folds_score])
    confusion_matrix_avg = np.mean([fold["confusion_matrix"] for fold in folds_score], axis=0)

    reports = [fold["classification_report"] for fold in folds_score]
    reports_avg = pd.concat(reports).groupby(level=0).mean()

    return {
        "f1_binary": f1_binary_avg,
        "f1_macro": f1_macro_avg,
        "competition_evaluation": competition_avg,
        "confusion_matrix": confusion_matrix_avg,
        "classification_report": reports_avg,
    }