import pandas as pd
import os
import joblib
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

FEATURES = ["ordered_before", "abandoned_before", "global_popularity", "set_as_regular"]

TARGET = "outcome"

LOCAL_DIRECTORY = "/mnt/d/zrive-ds/src/module_2/files_downloaded"
FINAL_DIRECTORY = "/mnt/d/zrive-ds/src/module_3/mvps"

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

    model_linear = make_pipeline(
        StandardScaler(),
        LinearRegression()
    ) 
    model_linear.fit(X, y)

    save_model(model_linear, "Linear")

def main():
    feature_frame = load_data(5)
    modeling(feature_frame)

if __name__ == "__main__": 

    main()