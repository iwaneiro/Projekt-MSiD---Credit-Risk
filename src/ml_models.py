from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import config

def get_all_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, random_state=config.RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=config.RANDOM_SEED),
        "Linear Regression": LinearRegression()
    }