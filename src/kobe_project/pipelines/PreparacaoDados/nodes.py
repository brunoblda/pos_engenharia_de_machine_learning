"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.12
"""

import pandas as pd
from pycaret.classification import ClassificationExperiment, setup, get_config
import mlflow

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df[["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]]
    df.dropna(inplace=True)
    df["shot_made_flag"] = df["shot_made_flag"].astype(int)
    return df

def prepare_train_and_test_data(df: pd.DataFrame, session_id: int, test_size: float) -> tuple:
    
    train_size = 1 - test_size
    
    setup(
        data=df,
        target="shot_made_flag",
        train_size=train_size,
        session_id=session_id,
        fold_shuffle=True,
        )

    X_train = get_config("X_train")
    y_train = get_config("y_train")
    X_test = get_config("X_test")
    y_test = get_config("y_test")
    
    train_df = X_train.copy()
    train_df["shot_made_flag"] = y_train.values

    test_df = X_test.copy()
    test_df["shot_made_flag"] = y_test.values

    with mlflow.start_run(run_name="pre_process", nested=True):
        mlflow.log_param("test_size_percent", test_size * 100)
        mlflow.log_metric("train_size", train_df.shape[0])
        mlflow.log_metric("test_size", test_df.shape[0])
    
    return train_df, test_df

