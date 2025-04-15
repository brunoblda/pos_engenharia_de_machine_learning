import pandas as pd
from pycaret.classification import (
    ClassificationExperiment, 
    predict_model
)
from sklearn.metrics import log_loss, f1_score, accuracy_score, roc_auc_score
import mlflow
from datetime import datetime

def train_logistic_regression(train_df: pd.DataFrame, session_id: int):
    with mlflow.start_run(run_name="logistic_regression", nested=True):
        exp = ClassificationExperiment()
        exp.setup(
            data=train_df,
            target="shot_made_flag",
            session_id=session_id,
            verbose=False
        )

        base_model = exp.create_model('lr')
        tuned_model = exp.tune_model(base_model, n_iter=20, optimize="AUC")
        final_model = exp.finalize_model(tuned_model)

        metrics_df = exp.pull()  # após tune_model() ou create_model()
        metrics_dict = metrics_df.iloc[0].to_dict()

        # Loga cada métrica no MLflow
        for key, value in metrics_dict.items():
            if isinstance(value, (float, int)):  # Evita strings como o nome do modelo
                mlflow.log_metric(key.lower(), value)

        # Loga parâmetros
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("session_id", session_id)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        hiperparams = final_model.get_params()
        mlflow.log_params(hiperparams)

        # Loga o modelo como artefato
        mlflow.sklearn.log_model(final_model, artifact_path="logistic_model")

        return final_model

def train_decision_tree(train_df: pd.DataFrame, session_id: int):
    with mlflow.start_run(run_name="decision_tree", nested=True):
        exp = ClassificationExperiment()
        exp.setup(
            data=train_df,
            target="shot_made_flag",
            session_id=session_id,
            verbose=False
        )

        base_model = exp.create_model('dt')
        tuned_model = exp.tune_model(base_model, n_iter=20, optimize='AUC')
        final_model = exp.finalize_model(tuned_model)
        
        metrics_df = exp.pull()  # após tune_model() ou create_model()
        metrics_dict = metrics_df.iloc[0].to_dict()

        # Loga cada métrica no MLflow
        for key, value in metrics_dict.items():
            if isinstance(value, (float, int)):  # Evita strings como o nome do modelo
                mlflow.log_metric(key.lower(), value)
        

        mlflow.log_param("model", "decision_tree")
        mlflow.log_param("session_id", session_id)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        hiperparams = final_model.get_params()
        mlflow.log_params(hiperparams)


        mlflow.sklearn.log_model(final_model, artifact_path="tree_model")

        return final_model 

def train_best_model(train_df: pd.DataFrame, session_id: int):
    with mlflow.start_run(run_name="best_model", nested=True):
        exp = ClassificationExperiment()
        exp.setup(
            data=train_df,
            target="shot_made_flag",
            session_id=session_id,
            verbose=False
        )

        best_model = exp.compare_models(n_select=1)
        tuned_model = exp.tune_model(best_model, n_iter=20, optimize='AUC')
        final_model = exp.finalize_model(tuned_model)

        metrics_df = exp.pull()  # após tune_model() ou create_model()
        metrics_dict = metrics_df.iloc[0].to_dict()

        # Loga cada métrica no MLflow
        for key, value in metrics_dict.items():
            if isinstance(value, (float, int)):  # Evita strings como o nome do modelo
                mlflow.log_metric(key.lower(), value)
        

        mlflow.log_param("model", "best_model (compare_models)")
        mlflow.log_param("session_id", session_id)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        hiperparams = final_model.get_params()
        mlflow.log_params(hiperparams)

        mlflow.sklearn.log_model(final_model, artifact_path="best_model")

        return final_model 

def evaluate_and_log_models(lr_model, dt_model, best_model, test_df: pd.DataFrame):
    with mlflow.start_run(run_name="model_evaluation", nested=True):

        y_test = test_df["shot_made_flag"]

        # Predições com Logistic Regression
        lr_predictions = predict_model(lr_model, data=test_df, raw_score=True)
        lr_probas = lr_predictions["prediction_score_1"]
        lr_preds = lr_predictions["prediction_label"]

        # Logistic Regression - Log Loss, F1 Score e Acurácia
        lr_logloss = log_loss(y_test, lr_probas)
        lr_f1 = f1_score(y_test, lr_preds)
        lr_accuracy = accuracy_score(y_test, lr_preds)
        lr_roc_auc = roc_auc_score(y_test, lr_probas)

        # Predições com Decision Tree
        dt_predictions = predict_model(dt_model, data=test_df, raw_score=True)
        dt_probas = dt_predictions["prediction_score_1"]
        dt_preds = dt_predictions["prediction_label"]

        # Decision Tree - Log Loss, F1 Score e Acurácia
        dt_logloss = log_loss(y_test, dt_probas)
        dt_f1 = f1_score(y_test, dt_preds)
        dt_accuracy = accuracy_score(y_test, dt_preds)
        dt_roc_auc = roc_auc_score(y_test, dt_probas)

        # Predições com Best Model
        best_predictions = predict_model(best_model, data=test_df, raw_score=True)
        best_probas = best_predictions["prediction_score_1"]
        best_preds = best_predictions["prediction_label"]

        # Best model - Log Loss, F1 Score e Acurácia
        best_logloss = log_loss(y_test, best_probas)
        best_f1 = f1_score(y_test, best_preds)
        best_accuracy = accuracy_score(y_test, best_preds)
        best_roc_auc = roc_auc_score(y_test, best_probas)

        # Métricas - Logistic Regression 
        mlflow.log_metric("lr_log_loss", lr_logloss)
        mlflow.log_metric("lr_f1_score", lr_f1)
        mlflow.log_metric("lr_accuracy", lr_accuracy)
        mlflow.log_metric("lr_roc_auc", lr_roc_auc)

        # Métricas - Decision Tree
        mlflow.log_metric("dt_log_loss", dt_logloss)
        mlflow.log_metric("dt_f1_score", dt_f1)
        mlflow.log_metric("dt_accuracy", dt_accuracy)
        mlflow.log_metric("dt_roc_auc", dt_roc_auc)

        # Métricas - Melhor Modelo
        mlflow.log_metric("best_log_loss", best_logloss)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_metric("best_roc_auc", best_roc_auc)
 
 