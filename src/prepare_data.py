import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import config
import os
import joblib


def load_and_prepare_data(apply_smote=False):
    df = pd.read_csv(config.DATA_PATH)

    column_mapping = {
        'laufkont': 'checking_status',
        'laufzeit': 'duration',
        'moral': 'credit_history',
        'verw': 'purpose',
        'hoehe': 'credit_amount',
        'sparkont': 'savings_status',
        'beszeit': 'employment_duration',
        'rate': 'installment_rate',
        'famges': 'personal_status_sex',
        'buerge': 'other_debtors',
        'wohnzeit': 'present_residence_since',
        'verm': 'property',
        'alter': 'age',
        'weitkred': 'other_installment_plans',
        'wohn': 'housing',
        'bishkred': 'number_credits',
        'beruf': 'job',
        'pers': 'people_liable',
        'telef': 'telephone',
        'gastarb': 'foreign_worker',
        'kredit': config.TARGET_COL
    }

    df.rename(columns=column_mapping, inplace=True)

    X = df[config.SELECTED_FEATURES]
    y = df[config.TARGET_COL]

    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=config.RANDOM_SEED)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=feature_names).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=feature_names).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    joblib.dump(scaler, "data/processed/standard_scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names