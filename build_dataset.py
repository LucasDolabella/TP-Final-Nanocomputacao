from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
import joblib

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
BASE_DIR = Path(__file__).parent

DATASET_INPUT_DIR = BASE_DIR / "dataset" / "input"
DATASET_OUTPUT_DIR = BASE_DIR / "dataset" / "output"

GEOM_DIR = BASE_DIR / "results" / "geometries"
PREP_DIR = BASE_DIR / "results" / "prepared"
PREP_DIR.mkdir(parents=True, exist_ok=True)

# Estruturas presentes (pulando a 06, que não existe)
STRUCTURES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ANGLES = [0, 15, 30, 45, 60, 75]

IMG_SIZE = 128
N_PCA = 20


def load_spectrum(struct_id: int, angle_deg: float):
    """
    Lê o arquivo Output_XX.txt da estrutura e retorna
    (wavelengths, field) apenas para o ângulo desejado.
    Formato esperado: col0=ângulo, col1=lambda, col2=campo.
    Linhas começando com '%' são ignoradas (comentários/cabeçalho).
    """
    out_path = DATASET_OUTPUT_DIR / f"Output_{struct_id:02d}.txt"

    # Ignora linhas iniciadas com '%' (comentários)
    data = np.loadtxt(out_path, comments="%", dtype=float)

    # Se só tiver uma linha válida, garante que fique 2D
    if data.ndim == 1:
        data = data[np.newaxis, :]

    subset = data[data[:, 0] == angle_deg]
    lambdas = subset[:, 1]
    field = subset[:, 2]
    return lambdas, field


def load_geometry_image(struct_id: int, angle_deg: float):
    """
    Lê a imagem geom_XX_ang.png e retorna array (128,128,1)
    com valores em [0,1].
    """
    img_path = GEOM_DIR / f"geom_{struct_id:02d}_{angle_deg}.png"
    img = imread(img_path)

    # Se tiver canais (RGB/RGBA), pega apenas um
    if img.ndim == 3:
        img = img[..., 0]

    # Garante tamanho 128x128 (caso não esteja)
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img = img.astype(np.float32)

    # Normaliza para [0,1]
    max_val = img.max()
    if max_val > 0:
        img = img / max_val

    # Adiciona canal (H,W,1)
    img = img[..., np.newaxis]
    return img


def main():
    X_list = []
    Y_list = []
    meta_list = []
    y_min_list = []
    y_max_list = []
    wavelengths = None

    # ==============================
    # 1. Carregar imagens + curvas
    # ==============================
    for struct_id in STRUCTURES:
        for ang in ANGLES:
            # --- imagem da geometria ---
            img = load_geometry_image(struct_id, ang)
            X_list.append(img)

            # --- curva de absorção ---
            lambdas, field = load_spectrum(struct_id, ang)

            if wavelengths is None:
                wavelengths = lambdas

            Y_list.append(field)
            meta_list.append([struct_id, ang])

            y_min_list.append(field.min())
            y_max_list.append(field.max())

            print(f"Carregado: estrutura {struct_id:02d}, {ang}°")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,128,128,1)
    Y_raw = np.stack(Y_list, axis=0).astype(np.float32)  # (N,101)
    metadata = np.array(meta_list, dtype=np.float32)  # (N,2)
    curve_min = np.array(y_min_list, dtype=np.float32)  # (N,)
    curve_max = np.array(y_max_list, dtype=np.float32)  # (N,)
    wavelengths = np.array(wavelengths, dtype=np.float32)  # (101,)

    print("\nShapes:")
    print("X:", X.shape)
    print("Y_raw:", Y_raw.shape)
    print("metadata:", metadata.shape)

    # =========================================
    # 2. Normalizar curvas (min-max por curva)
    # =========================================
    denom = curve_max - curve_min
    # Evita divisão por zero
    denom[denom == 0] = 1.0

    Y_norm = (Y_raw - curve_min[:, None]) / denom[:, None]

    # ==========================
    # 3. PCA nas curvas normalizadas
    # ==========================
    print("\nCalculando PCA nas curvas normalizadas...")
    pca = PCA(n_components=N_PCA)
    Y_pca = pca.fit_transform(Y_norm)

    explained = pca.explained_variance_ratio_ * 100.0
    print("Y_pca shape:", Y_pca.shape)
    print("Variância explicada (%):", np.round(explained, 3))
    print("Variância total explicada: {:.3f} %".format(explained.sum()))

    # ==========================
    # 4. Salvar tudo
    # ==========================
    np.save(PREP_DIR / "X_images.npy", X)
    np.save(PREP_DIR / "Y_raw.npy", Y_raw)
    np.save(PREP_DIR / "Y_pca.npy", Y_pca)
    np.save(PREP_DIR / "wavelengths.npy", wavelengths)
    np.save(PREP_DIR / "metadata.npy", metadata)
    np.save(PREP_DIR / "curve_min.npy", curve_min)
    np.save(PREP_DIR / "curve_max.npy", curve_max)

    joblib.dump(pca, PREP_DIR / "pca_model.pkl")

    print("\nArquivos salvos em:", PREP_DIR.resolve())


if __name__ == "__main__":
    main()
