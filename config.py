RANDOM_SEED = 42
TEST_SIZE = 0.2
DATA_PATH = "data/raw/german_credit_data.csv"

SELECTED_FEATURES = [
    'checking_status',
    'duration',
    'credit_history',
    'purpose',
    'credit_amount',
    'savings_status',
    'employment_duration',
    'installment_rate',
    'personal_status_sex',
    'other_debtors',
    'present_residence_since',
    'property',
    'age',
    'other_installment_plans',
    'housing',
    'number_credits',
    'job',
    'people_liable',
    'telephone',
    'foreign_worker'
]

TARGET_COL = 'credit_risk'

RF_N_ESTIMATORS = 100
LR_MAX_ITER = 1000