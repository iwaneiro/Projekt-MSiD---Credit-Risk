import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_test, y_test, is_continuous=False):
    start_time = time.time()

    predictions = model.predict(X_test)

    if is_continuous:
        predictions = [1 if p >= 0.5 else 0 for p in predictions]

    end_time = time.time()
    execution_time = end_time - start_time

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    return {
        'time_seconds': execution_time,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': predictions
    }

def evaluate_fuzzy_system(fuzzy_sys, X_test_scaled, y_test, scaler, feature_names, num_attributes=2):
    start_time = time.time()

    X_test_raw = scaler.inverse_transform(X_test_scaled)

    idx_age = feature_names.index('age')
    idx_amount = feature_names.index('credit_amount')
    if num_attributes >= 3:
        idx_duration = feature_names.index('duration')

    predictions = []

    for row in X_test_raw:
        age_val = row[idx_age]
        amount_val = row[idx_amount]

        age_val = np.clip(age_val, 18, 100)
        amount_val = np.clip(amount_val, 200, 20000)

        # Wyliczenie ryzyka
        if num_attributes >= 3:
            duration_val = np.clip(row[idx_duration], 4, 72)
            risk = fuzzy_sys.predict(age_val, amount_val, duration_val)
        else:
            risk = fuzzy_sys.predict(age_val, amount_val)

        if risk is not None and risk <= 50.0:
            predictions.append(1)
        else:
            predictions.append(0)

    end_time = time.time()
    execution_time = end_time - start_time

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    return {
        'time_seconds': execution_time,
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': predictions
    }