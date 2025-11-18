from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# =========================
# CARREGAR DADOS + MODELO
# =========================
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"
MODELS_DIR = BASE_DIR / "results" / "models"
COMP_DIR = BASE_DIR / "results" / "comparisons"
COMP_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(PREP_DIR / "X_images.npy")  # (N,128,128,1)
Y_raw = np.load(PREP_DIR / "Y_raw.npy")  # (N,101)
metadata = np.load(PREP_DIR / "metadata.npy")
wavelengths = np.load(PREP_DIR / "wavelengths.npy")
curve_min = np.load(PREP_DIR / "curve_min.npy")
curve_max = np.load(PREP_DIR / "curve_max.npy")

pca = joblib.load(PREP_DIR / "pca_model.pkl")
model = tf.keras.models.load_model(MODELS_DIR / "cnn_pca_model.keras")

print("X:", X.shape)
print("Y_raw:", Y_raw.shape)
print("metadata:", metadata.shape)


def plot_example(idx: int):
    """
    Faz a predição para o índice idx, reconstrói a curva
    e plota SIM (curva original) vs DL (modelo).
    """
    x_img = X[idx : idx + 1]
    y_true = Y_raw[idx]
    y_min = curve_min[idx]
    y_max = curve_max[idx]

    # 1) Predição nos componentes principais
    y_pca_pred = model.predict(x_img, verbose=0)[0]  # (20,)

    # 2) Reconstrói curva normalizada via PCA inverso
    y_norm_pred = pca.inverse_transform(y_pca_pred.reshape(1, -1))[0]

    # 3) Desnormaliza para a escala original
    y_pred = y_norm_pred * (y_max - y_min) + y_min

    struct_id = int(metadata[idx, 0])
    angle = int(metadata[idx, 1])

    plt.figure(figsize=(6, 4))
    plt.plot(wavelengths, y_true, label="SIM")
    plt.plot(wavelengths, y_pred, label="DL")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption (a.u.)")
    plt.title("Absorption")
    plt.grid(True)
    plt.legend()

    out_path = COMP_DIR / f"comparison_{struct_id:02d}_{angle}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {out_path}")


if __name__ == "__main__":
    # Escolhe alguns exemplos (pode ajustar depois)
    exemplos = [0, 5, 20, 35, 50]
    for idx in exemplos:
        if idx < len(X):
            plot_example(idx)
