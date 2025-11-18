# analyze_results.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# =========================
# CONFIGURAÇÕES
# =========================
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"
MODELS_DIR = BASE_DIR / "results" / "models"
ANALYSIS_DIR = BASE_DIR / "results" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. PLOTAR HISTÓRICO DE TREINO
# =========================
def plot_history():
    hist_path = MODELS_DIR / "history.npy"
    if not hist_path.exists():
        print(f"[HIST] Arquivo {hist_path} não encontrado. Pulei essa parte.")
        return

    history = np.load(hist_path, allow_pickle=True).item()

    epochs = np.arange(1, len(history["loss"]) + 1)

    # --- Loss ---
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

    # --- MAE ---
    if "mae" in history and "val_mae" in history:
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
    # Carregar dados necessários
    X = np.load(PREP_DIR / "X_images.npy")  # (N, 128, 128, 1)
    Y_raw = np.load(PREP_DIR / "Y_raw.npy")  # (N, 101)
    wavelengths = np.load(PREP_DIR / "wavelengths.npy")  # (101,)
    curve_min = np.load(PREP_DIR / "curve_min.npy")  # (N,)
    curve_max = np.load(PREP_DIR / "curve_max.npy")  # (N,)

    pca = joblib.load(PREP_DIR / "pca_model.pkl")
    model = tf.keras.models.load_model(MODELS_DIR / "cnn_pca_model.keras")

    print("[MAE λ] Gerando predições para todo o conjunto...")
    # 1) Predição nos componentes principais
    Y_pca_pred = model.predict(X, batch_size=8, verbose=0)  # (N, n_components)

    # 2) Reconstruir curvas normalizadas via PCA inverso
    Y_norm_pred = pca.inverse_transform(Y_pca_pred)  # (N, 101)

    # 3) Desnormalizar para escala original
    denom = (curve_max - curve_min)[:, None]  # (N,1)
    denom[denom == 0] = 1.0
    Y_pred = Y_norm_pred * denom + curve_min[:, None]  # (N, 101)

    # 4) Erro absoluto por ponto (amostra x comprimento de onda)
    errors = np.abs(Y_pred - Y_raw)  # (N, 101)

    # 5) MAE global e MAE por comprimento de onda
    mae_global = errors.mean()
    mae_por_lambda = errors.mean(axis=0)  # (101,)

    print(f"[MAE λ] MAE global médio em toda a curva: {mae_global:.6f}")

    # 6) Plot MAE por comprimento de onda
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


if __name__ == "__main__":
    print("=== Análise do modelo ===")
    plot_history()
    mae_por_lambda()
    print("=== Fim da análise ===")
