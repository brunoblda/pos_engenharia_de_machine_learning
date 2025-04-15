"""
This is a boilerplate pipeline 'Aplicacao'
generated using Kedro 0.19.12
"""

import mlflow
import pandas as pd

from pycaret.classification import (
    predict_model
)
from sklearn.metrics import log_loss, f1_score, accuracy_score, roc_auc_score

def model_prod(model, prod_df: pd.DataFrame):

    with mlflow.start_run(run_name="PipelineAplicação", nested=True):

        # Loga os parâmetros do modelo
        hiperparams = model.get_params()
        mlflow.log_params(hiperparams)

        # Loga o modelo como artefato
        mlflow.sklearn.log_model(model, artifact_path="model_prod")

        # Registra o modelo como ofícial
        mlflow.register_model(
            model_uri=mlflow.get_artifact_uri("model_prod"),
            name="model_prod"
        )

        y_test = prod_df["shot_made_flag"]

        # Predições 
        predictions = predict_model(model, data=prod_df, raw_score=True)
        probas = predictions["prediction_score_1"]
        preds = predictions["prediction_label"]

        # Salva resultados
        predictions.to_parquet("predicoes_producao.parquet", index=False)
        mlflow.log_artifact("predicoes_producao.parquet")

        # Log Loss, F1 Score e Acurácia
        logloss = log_loss(y_test, probas)
        f1 = f1_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probas)

        # Métricas 
        mlflow.log_metric("model_prod_log_loss", logloss)
        mlflow.log_metric("model_prod_f1_score", f1)
        mlflow.log_metric("model_prod_accuracy", accuracy)
        mlflow.log_metric("model_prod_roc_auc", roc_auc)

        mlflow.log_metric("prod_size", prod_df.shape[0])

        return predictions, model 

