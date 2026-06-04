import shap
import matplotlib.pyplot as plt
import os
import numpy as np


def generate_shap_summary(model, X_train, feature_names):
    plt.clf()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    fig = plt.figure(figsize=(10, 8))

    shap.summary_plot(
        vals,
        X_train,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()