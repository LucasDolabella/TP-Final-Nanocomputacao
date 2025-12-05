# analyze_results.py
"""
Script de Análise de Resultados do Modelo de Deep Learning

Este script realiza a análise pós-treinamento do modelo CNN+PCA, incluindo:
1. Visualização do histórico de treino (loss e MAE ao longo das épocas)
2. Cálculo e visualização do erro médio absoluto (MAE) por comprimento de onda

O script carrega o modelo treinado, realiza predições em todo o dataset e
gera gráficos para avaliar o desempenho do modelo.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# =========================
# CONFIGURAÇÕES
# =========================
# Define os diretórios do projeto
BASE_DIR = Path(__file__).parent  # Diretório raiz do projeto
PREP_DIR = BASE_DIR / "results" / "prepared"  # Dados preparados (arrays numpy, PCA)
MODELS_DIR = BASE_DIR / "results" / "models"  # Modelos treinados e histórico
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"  # Destino para gráficos de análise
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)  # Cria diretório se não existir


# =========================
# 1. PLOTAR HISTÓRICO DE TREINO
# =========================
def plot_history():
    """
    Plota o histórico de treinamento do modelo.
    
    Gera dois gráficos:
    - Loss (MSE) ao longo das épocas para treino e validação
    - MAE ao longo das épocas para treino e validação (se disponível)
    
    Os gráficos são salvos no diretório ANALYSIS_DIR.
    """
    hist_path = MODELS_DIR / "history.npy"
    if not hist_path.exists():
        print(f"[HIST] Arquivo {hist_path} não encontrado. Pulei essa parte.")
        return

    # Carrega o histórico de treino salvo durante o treinamento
    history = np.load(hist_path, allow_pickle=True).item()

    # Cria array com número de épocas para o eixo x
    epochs = np.arange(1, len(history["loss"]) + 1)

    # --- Loss (MSE) ---
    # Cria gráfico mostrando a evolução do erro quadrático médio
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["loss"], label="treino (loss)")
    plt.plot(epochs, history["val_loss"], label="validação (loss)")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Histórico de treino - Loss")
    plt.grid(True)
    plt.legend()
    out_path = ANALYSIS_DIR / "history_loss.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[HIST] Figura salva: {out_path}")

    # --- MAE (Erro Médio Absoluto) ---
    # Verifica se o histórico contém métricas de MAE
    if "mae" in history and "val_mae" in history:
        # Cria gráfico mostrando a evolução do erro médio absoluto
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["mae"], label="treino (MAE)")
        plt.plot(epochs, history["val_mae"], label="validação (MAE)")
        plt.xlabel("Época")
        plt.ylabel("MAE")
        plt.title("Histórico de treino - MAE")
        plt.grid(True)
        plt.legend()
        out_path = ANALYSIS_DIR / "history_mae.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[HIST] Figura salva: {out_path}")
    else:
        print("[HIST] Chaves 'mae'/'val_mae' não encontradas no histórico.")


# =========================
# 2. MAE POR COMPRIMENTO DE ONDA
# =========================
def mae_por_lambda():
    """
    Calcula e plota o MAE (Mean Absolute Error) por comprimento de onda.
    
    Este análise permite identificar em quais comprimentos de onda o modelo
    tem maior ou menor erro de predição. O processo inclui:
    1. Carregar dados e modelo
    2. Gerar predições para todo o dataset
    3. Reconstruir curvas a partir dos componentes PCA preditos
    4. Desnormalizar para escala original
    5. Calcular erros absolutos
    6. Plotar MAE médio por comprimento de onda
    """
    # Carregar todos os dados necessários para a análise
    X = np.load(PREP_DIR / "X_images.npy")  # Imagens de geometrias (N, 128, 128, 1)
    Y_raw = np.load(PREP_DIR / "Y_raw.npy")  # Curvas de absorção originais (N, 101)
    wavelengths = np.load(PREP_DIR / "wavelengths.npy")  # Comprimentos de onda (101,)
    curve_min = np.load(PREP_DIR / "curve_min.npy")  # Valores mínimos para desnormalização (N,)
    curve_max = np.load(PREP_DIR / "curve_max.npy")  # Valores máximos para desnormalização (N,)res máximos para desnormalização (N,)

    # Carregar modelo PCA (para transformação inversa) e modelo CNN treinado
    pca = joblib.load(PREP_DIR / "pca_model.pkl")
    model = tf.keras.models.load_model(MODELS_DIR / "cnn_pca_model.keras")

    print("[MAE λ] Gerando predições para todo o conjunto...")
    
    # Passo 1: Predição nos componentes principais PCA
    # O modelo CNN prediz os componentes PCA a partir das imagens de geometria
    Y_pca_pred = model.predict(X, batch_size=8, verbose=0)  # (N, n_components)

    # Passo 2: Reconstruir curvas normalizadas via transformação inversa do PCA
    # Converte os componentes PCA de volta para o espaço original de 101 pontos
    Y_norm_pred = pca.inverse_transform(Y_pca_pred)  # (N, 101)

    # Passo 3: Desnormalizar para escala original
    # Reverte a normalização min-max aplicada durante o pré-processamento
    denom = (curve_max - curve_min)[:, None]  # (N, 1) - denominador para desnormalização
    denom[denom == 0] = 1.0  # Evita divisão por zero para curvas constantes
    Y_pred = Y_norm_pred * denom + curve_min[:, None]  # (N, 101) - curvas preditas em escala original - curvas preditas em escala original

    # Passo 4: Calcular erro absoluto por ponto
    # Matriz de erros onde cada elemento [i,j] é o erro absoluto da amostra i no comprimento de onda j
    errors = np.abs(Y_pred - Y_raw)  # (N, 101)

    # Passo 5: Calcular MAE global e MAE por comprimento de onda
    mae_global = errors.mean()  # MAE médio considerando todas as amostras e todos os λ
    mae_por_lambda = errors.mean(axis=0)  # MAE médio para cada λ (média sobre todas as amostras)

    print(f"[MAE λ] MAE global médio em toda a curva: {mae_global:.6f}")

    # Passo 6: Plotar MAE por comprimento de onda
    # Este gráfico mostra em quais regiões espectrais o modelo é mais/menos preciso
    plt.figure(figsize=(6, 4))
    plt.plot(wavelengths, mae_por_lambda)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MAE da absorção")
    plt.title("MAE por comprimento de onda")
    plt.grid(True)
    out_path = ANALYSIS_DIR / "mae_by_wavelength.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[MAE λ] Figura salva: {out_path}")


# =========================
# EXECUÇÃO PRINCIPAL
# =========================
if __name__ == "__main__":
    """
    Executa todas as análises do modelo:
    1. Plota histórico de treino (loss e MAE ao longo das épocas)
    2. Calcula e plota MAE por comprimento de onda
    
    Os gráficos gerados são salvos em results/analysis/
    """
    print("=== Análise do modelo ===")
    plot_history()  # Gera gráficos de histórico de treino
    mae_por_lambda()  # Gera análise de erro por comprimento de onda
    print("=== Fim da análise ===")