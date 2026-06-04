import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        features_sorted = [feature_names[i] for i in indices]

        plt.figure(figsize=(8, 5))

        sns.barplot(
            x=importances[indices],
            y=features_sorted,
            hue=features_sorted,
            palette='viridis',
            legend=False
        )

        plt.title(f'Feature Importances: {model_name}')
        plt.tight_layout()
        plt.show()


def plot_benchmark_results(results_df):
    """
    Generuje zbiorcze wykresy słupkowe porównujące Accuracy i Czas wykonania.
    Zapisuje obrazek do folderu data/processed/ i wyświetla go na ekranie.
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Dokładność (Accuracy) ---
    sns.barplot(
        data=results_df,
        x='Accuracy',
        y='Model / Konfiguracja',
        hue='Metodologia',
        ax=axes[0],
        palette='Set1'
    )
    axes[0].set_title('Porównanie dokładności (Accuracy)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Skuteczność (0.0 - 1.0)', fontsize=12)
    axes[0].set_ylabel('')
    axes[0].set_xlim(0, 1.0)

    # --- Czas wykonania (Time) ---
    sns.barplot(
        data=results_df,
        x='Czas (s)',
        y='Model / Konfiguracja',
        hue='Metodologia',
        ax=axes[1],
        palette='Set1'
    )
    axes[1].set_title('Porównanie czasu wykonania', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Czas (sekundy)', fontsize=12)
    axes[1].set_ylabel('')

    axes[1].get_legend().remove()

    plt.tight_layout()

    os.makedirs("data/processed", exist_ok=True)
    plot_path = "data/processed/benchmark_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"\n[Wykresy] Zapisano podsumowanie do pliku: {plot_path}")

    plt.show()


def plot_confusion_matrices_comparison(y_test, best_ml_preds, best_fuzzy_preds,
                                       ml_name="Random Forest", fuzzy_name="3 atrybuty, 3x trimf"):
    """
    Rysuje macierze pomyłek dla najlepszego modelu ML i najlepszego systemu rozmytego
    obok siebie. Zapisuje do data/processed/confusion_matrices.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Macierze pomyłek – porównanie najlepszych modeli', fontsize=15, fontweight='bold')

    pairs = [
        (best_ml_preds, ml_name, axes[0]),
        (best_fuzzy_preds, fuzzy_name, axes[1]),
    ]

    for preds, name, ax in pairs:
        cm = confusion_matrix(y_test, preds)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: Zły (0)', 'Pred: Dobry (1)'],
            yticklabels=['True: Zły (0)', 'True: Dobry (1)']
        )
        ax.set_title(
            f'{name}\nPrecision: {precision:.3f}  |  Recall: {recall:.3f}',
            fontsize=11
        )
        ax.set_ylabel('Prawdziwa klasa')
        ax.set_xlabel('Przewidziana klasa')

    plt.tight_layout()
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Wykresy] Zapisano macierze pomyłek do: {save_path}")
    plt.show()