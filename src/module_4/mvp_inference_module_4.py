import pandas as pd
import os
from joblib import load
from mvp_code_module_4 import FINAL_DIRECTORY, FEATURES, TARGET, load_data

def save_preds(data: pd.Series, model_name: str):
    data.to_csv(os.path.join(FINAL_DIRECTORY, f"predictions_{model_name}"))


def main():
    model_name = "20250224-231839_Non-Linear-Random-Forest.pkl"
    model = load(os.path.join(FINAL_DIRECTORY, model_name))

    df = load_data(5)
    X = df[FEATURES]
    y_pred = model.predict(X) 
    y_pred_df = pd.DataFrame(y_pred, columns=["Predictions"])
    save_preds(y_pred_df, "Non_Linear")
    

if __name__ == "__main__":
    main()