import joblib
import pandas as pd
from src.prepare_data import load_and_prepare_data
from src.evaluate import evaluate_model, evaluate_fuzzy_system
from src.ml_models import get_all_models
from src.fuzzy_logic_system import CreditRiskFuzzySystem
from src.tune import tune_random_forest
from src.plots import plot_benchmark_results, plot_confusion_matrices_comparison
from src.xai import generate_shap_summary

def main():
    print("Ładowanie i przygotowanie danych...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data(apply_smote=True)

    all_results = []
    best_ml_predictions = None

    print("\n" + "=" * 50)
    print("TESTOWANIE MODELI MACHINE LEARNING")
    print("=" * 50)

    models = get_all_models()

    for model_name, model in models.items():
        print(f"Trenowanie i ocena: {model_name}...")

        if model_name == "Random Forest":
            trained_model = tune_random_forest(X_train, y_train)
            models[model_name] = trained_model
        else:
            trained_model = model
            trained_model.fit(X_train, y_train)

        is_cont = True if model_name == "Linear Regression" else False
        metrics = evaluate_model(trained_model, X_test, y_test, is_continuous=is_cont)

        if model_name == "Random Forest":
            best_ml_predictions = metrics['predictions']

        all_results.append({
            'Metodologia': 'Machine Learning',
            'Model / Konfiguracja': model_name,
            'Czas (s)': round(metrics['time_seconds'], 4),
            'Accuracy': round(metrics['accuracy'], 4),
            'F1 Score': round(metrics['f1_score'], 4)
        })

    best_rf = models["Random Forest"]
    joblib.dump(best_rf, "data/processed/best_rf_model.pkl")

    print("\nGenerowanie analizy SHAP dla najlepszego modelu ML (wszystkie 20 cech)...")
    generate_shap_summary(best_rf, X_train, feature_names)

    print("\n" + "=" * 50)
    print("TESTOWANIE SYSTEMÓW ROZMYTYCH (FUZZY LOGIC)")
    print("=" * 50)

    fuzzy_variants = [
        {"attr": 2, "mf": 3, "type": 'trimf', "name": '2 atrybuty, 3x trimf (Baza)'},
        {"attr": 2, "mf": 3, "type": 'gaussmf', "name": '2 atrybuty, 3x gaussmf'},
        {"attr": 3, "mf": 3, "type": 'trimf', "name": '3 atrybuty, 3x trimf'},
        {"attr": 2, "mf": 5, "type": 'trimf', "name": '2 atrybuty, 5x trimf'},
        {"attr": 2, "mf": 7, "type": 'trimf', "name": '2 atrybuty, 7x trimf'},
        {"attr": 2, "mf": 3, "type": 'trapmf', "name": '2 atrybuty, 3x trapmf'},
    ]

    best_fuzzy_predictions = None

    for variant in fuzzy_variants:
        print(f"Uruchamiam: Fuzzy ({variant['name']})...")
        fuzzy_sys = CreditRiskFuzzySystem(
            num_attributes=variant['attr'],
            num_mf=variant['mf'],
            mf_type=variant['type']
        )

        metrics_f = evaluate_fuzzy_system(
            fuzzy_sys, X_test, y_test, scaler, feature_names, num_attributes=variant['attr']
        )

        if variant['name'] == '3 atrybuty, 3x trimf':
            best_fuzzy_predictions = metrics_f['predictions']

        all_results.append({
            'Metodologia': 'Fuzzy Logic',
            'Model / Konfiguracja': variant['name'],
            'Czas (s)': round(metrics_f['time_seconds'], 4),
            'Accuracy': round(metrics_f['accuracy'], 4),
            'F1 Score': round(metrics_f['f1_score'], 4)
        })

    print("\n" + "=" * 50)
    print("PODSUMOWANIE EKSPERYMENTU")
    print("=" * 50)

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    print("\nGenerowanie wykresów podsumowujących...")
    plot_benchmark_results(results_df)

    if best_ml_predictions is not None and best_fuzzy_predictions is not None:
        print("\nGenerowanie macierzy pomyłek...")
        plot_confusion_matrices_comparison(
            y_test,
            best_ml_predictions,
            best_fuzzy_predictions,
            ml_name="Random Forest (tuned)",
            fuzzy_name="Fuzzy: 3 atrybuty, 3× trimf"
        )

    print("\nGotowe! Eksperyment zakończony.")

if __name__ == "__main__":
    main()