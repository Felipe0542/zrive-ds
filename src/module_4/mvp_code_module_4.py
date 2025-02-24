import pandas as pd
import os
import joblib
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

LABEL_COLS = [
    "vendor",
    "product_type"
    ]
NUMERIC_COLS = [
    'user_order_seq',
    'ordered_before',
    'abandoned_before', 
    'active_snoozed', 
    'set_as_regular',
    'normalised_price',
    'discount_pct',
    'global_popularity',
    'count_adults',
    'count_children',
    'count_babies',
    'count_pets',
    'people_ex_baby',
    'days_since_purchase_variant_id',
    'avg_days_to_buy_variant_id',
    'std_days_to_buy_variant_id',
    'days_since_purchase_product_type',
    'avg_days_to_buy_product_type',
    'std_days_to_buy_product_type'
    ]

FEATURES = LABEL_COLS + NUMERIC_COLS

TARGET = "outcome"

LOCAL_DIRECTORY = "/mnt/d/zrive-ds/src/module_2/files_downloaded"
FINAL_DIRECTORY = "/mnt/d/zrive-ds/src/module_4/mvps"

def load_data(min_prods: int = 5) -> pd.DataFrame:
    
    name = "feature_frame.csv"
    file_path = os.path.join(LOCAL_DIRECTORY, name)
    df = pd.read_csv(file_path)
    df_bought = df[df['outcome'] == 1]
    order_counts = df_bought.groupby('order_id').size()
    orders_with_min_or_more = order_counts[order_counts >= min_prods].index
    filtered_df = df[df['order_id'].isin(orders_with_min_or_more)]

    return filtered_df

def save_model(model: BaseEstimator, model_name: str) -> pd.DataFrame:

    if not os.path.exists(FINAL_DIRECTORY):
        os.makedirs(FINAL_DIRECTORY)
    
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    joblib.dump(model, os.path.join(FINAL_DIRECTORY, model_name))

def modeling(data: pd.DataFrame) -> None:
    
    X = data[FEATURES]
    y= data[TARGET]
    categorical_processor = OneHotEncoder(handle_unknown="ignore", max_categories=20)

    preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", NUMERIC_COLS),
            ("categorical", categorical_processor, LABEL_COLS),
        ])

    model_non_linear_RF = make_pipeline(
        preprocessor,
        StandardScaler(),
        RandomForestClassifier(n_estimators=150, max_depth=10)
    ) 

    model_non_linear_RF.fit(X, y)

    save_model(model_non_linear_RF, "Non-Linear-Random-Forest")

def main():
    feature_frame = load_data(5)
    modeling(feature_frame)

if __name__ == "__main__": 

    main()