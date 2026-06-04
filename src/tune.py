from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import config


def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=config.RANDOM_SEED)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Najlepsze parametry RF: {grid_search.best_params_}")
    return grid_search.best_estimator_