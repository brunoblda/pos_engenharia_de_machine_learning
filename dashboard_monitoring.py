import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
)
from pathlib import Path
import joblib
import requests
import json

# Configurações iniciais
st.set_page_config(page_title="Monitoramento do Modelo", layout="wide")
st.title("📊 Monitoramento do Modelo em Produção")

# Navegação por abas
tabs = st.tabs(["📈 Monitoramento", "🧪 Predição manual"])

with tabs[0]:
    # Escolher modelo 
    st.sidebar.header("🧠 Escolha o modelo")
    opcao_modelo = st.sidebar.radio("Modelo a ser usado para predição:", ["Servido via MLflow", "Modelo local (.pkl)"])

    # Caminho padrão
    def caminho_padrao():
        caminho = Path("data/08_reporting/model_metric_prod_report.parquet")
        if caminho.exists():
            return caminho
        return None

    # Leitura do arquivo
    st.sidebar.header("🔧 Fonte de Dados")
    usar_padrao = st.sidebar.checkbox("Usar dados de produção local", value=True)

    if usar_padrao and caminho_padrao():
        df = pd.read_parquet(caminho_padrao())
        st.sidebar.success("Dados de producao carregados com sucesso!")
    else:
        uploaded_file = st.sidebar.file_uploader("Ou carregue manualmente um arquivo (.parquet, .csv ou .pkl/ .pickle)", type=["parquet", "csv", "pkl", "pickle"])
        if uploaded_file:
            if uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".pkl", ".pickle")):
                df = pd.read_pickle(uploaded_file)
            else:
                st.error("Formato de arquivo não suportado.")
                st.stop()
            st.sidebar.success("Arquivo carregado com sucesso!")
        else:
            st.warning("Nenhum arquivo carregado.")
            st.stop()

    # Exibir preview dos dados
    st.subheader("🔍 Amostra dos dados")
    st.dataframe(df.head())

    # Fazer predição
    st.subheader("🔮 Previsão do modelo selecionado")
    if st.button("Executar predição no dataset carregado"):
        input_data = df.drop(columns=["shot_made_flag", "prediction_label", "prediction_score_0", "prediction_score_1"], errors="ignore")

        # Verifica se o modelo está servido via MLflow ou local
        if opcao_modelo == "Servido via MLflow":
            try:
                response = requests.post(
                    url="http://127.0.0.1:1234/invocations",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"dataframe_split": input_data.to_dict(orient="split")})
                )
                if response.status_code == 200:
                    predictions = response.json()
                    if isinstance(predictions, list):
                        df["prediction_label"] = predictions
                    elif isinstance(predictions, dict) and "predictions" in predictions:
                        df["prediction_label"] = predictions["predictions"]
                    else:
                        st.error("Formato inesperado da resposta do modelo servido.")
                else:
                    st.error(f"Erro na resposta do modelo servido: {response.text}")
                st.success("Predição com modelo servido realizada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao conectar com o modelo servido: {e}")
                
        # Modelo local
        else:  
            try:
                model = joblib.load("data/08_reporting/production_model.pkl")
                df["prediction_label"] = model.predict(input_data)
                if hasattr(model, "predict_proba"):
                    df["prediction_score_1"] = model.predict_proba(input_data)[:, 1]
                st.success("Predição com modelo local realizada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar modelo local: {e}")

    # Avalia se existe predição
    if "shot_made_flag" in df.columns and "prediction_label" in df.columns:
        y_true = df["shot_made_flag"].astype(int)
        y_pred = df["prediction_label"].astype(int)
        y_prob = df.get("prediction_score_1")

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logloss = log_loss(y_true, y_prob) if y_prob is not None else None
        roc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("F1 Score", f"{f1:.3f}")
        if logloss: col3.metric("Log Loss", f"{logloss:.3f}")
        if roc: col4.metric("ROC AUC", f"{roc:.3f}")

        # Matriz de confusão
        st.subheader("🔄 Matriz de Confusão")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        st.pyplot(fig)

    # Distribuição das probabilidades 
    if "prediction_score_1" in df.columns:
        st.subheader("📊 Distribuição das Probabilidades (classe 1)")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["prediction_score_1"], bins=20, kde=True, ax=ax2)
        ax2.set_title("Distribuição dos Scores de Classe Positiva")
        st.pyplot(fig2)

with tabs[1]:
    st.header("🧪 Predição com entrada manual")
    
    # Inputs para predição 
    shot_distance = st.slider("Distância do arremesso", 0, 30, 15)
    period = st.selectbox("Período", [1, 2, 3, 4])
    lat = st.number_input("Latitude (lat)", min_value=33.0, max_value=35.0, value=34.02, step=0.01)
    lon = st.number_input("Longitude (lon)", min_value=-120.0, max_value=-117.0, value=-118.25, step=0.01)
    minutes_remaining = st.slider("Minutos restantes", 0, 11, 5)
    playoffs = st.checkbox("É playoff?", value=False)

    sample_input = pd.DataFrame([{
        "shot_distance": shot_distance,
        "period": period,
        "lat": lat,
        "lon": lon,
        "minutes_remaining": minutes_remaining,
        "playoffs": int(playoffs)
    }])

    expected_cols = ["shot_distance", "period", "lat", "lon", "minutes_remaining", "playoffs"]
    sample_input = sample_input[expected_cols]

    if st.button("📤 Enviar para predição"):
        try:
            response = requests.post(
                url="http://127.0.0.1:1234/invocations",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"dataframe_split": sample_input.to_dict(orient="split")})
            )
            if response.status_code == 200:
                pred = response.json()
                st.success(f"✅ Resultado da predição: {pred}")
            else:
                st.error(f"Erro ao predizer: {response.text}")
        except Exception as e:
            st.error(f"Erro ao conectar com o modelo: {e}")
