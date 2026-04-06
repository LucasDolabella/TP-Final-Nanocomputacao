"""
Script de Comparação de Curvas Espectrais

Este script compara curvas espectrais preditas pelo modelo de deep learning
com as curvas simuladas (ground truth). Gera gráficos lado a lado mostrando:
- SIM: curva espectral original obtida por simulação
- DL: curva espectral predita pelo modelo CNN+PCA

O processo:
1. Carrega dados processados e modelo treinado
2. Para exemplos selecionados, faz predições
3. Reconstrói curvas a partir dos componentes PCA preditos
4. Gera gráficos de comparação SIM vs DL

Os gráficos são salvos em results/comparisons/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# =========================
# CARREGAR DADOS + MODELO
# =========================
# Define diretórios do projeto
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"  # Dados preparados
MODELS_DIR = BASE_DIR / "results" / "models"  # Modelo treinado
COMP_DIR = BASE_DIR / "results" / "comparisons"  # Gráficos de comparação
COMP_DIR.mkdir(parents=True, exist_ok=True)

# Carrega todos os dados necessários para fazer comparações
X = np.load(PREP_DIR / "X_images.npy")  # (N, 128, 128, 1) - imagens de geometrias
Y_raw = np.load(PREP_DIR / "Y_raw.npy")  # (N, 101) - curvas espectrais originais (ground truth)
metadata = np.load(PREP_DIR / "metadata.npy")  # (N, 2) - [struct_id, ângulo]
wavelengths = np.load(PREP_DIR / "wavelengths.npy")  # (101,) - comprimentos de onda (eixo x)
curve_min = np.load(PREP_DIR / "curve_min.npy")  # (N,) - mínimo de cada curva (para desnormalização)
curve_max = np.load(PREP_DIR / "curve_max.npy")  # (N,) - máximo de cada curva (para desnormalização)

# Carrega modelo PCA (para transformação inversa) e modelo CNN treinado
pca = joblib.load(PREP_DIR / "pca_model.pkl")  # Modelo PCA para reconstruir curvas
model = tf.keras.models.load_model(MODELS_DIR / "cnn_pca_model.keras")  # Modelo CNN+PCA treinado  # Modelo CNN+PCA treinado

# Exibe informações sobre os dados carregados
print("X:", X.shape)
print("Y_raw:", Y_raw.shape)
print("metadata:", metadata.shape)


# =========================
# FUNÇÃO DE PLOTAGEM
# =========================
def plot_example(idx: int):
    """
    Gera comparação visual entre curva simulada (SIM) e predita (DL).
    
    Args:
        idx: Índice da amostra no dataset (0 a N-1)
    
    Processo:
    1. Seleciona a amostra (imagem de geometria e curva real)
    2. Faz predição usando o modelo CNN
    3. Reconstrói curva completa a partir dos componentes PCA preditos
    4. Desnormaliza para escala original
    5. Plota SIM vs DL e salva gráfico
    
    O gráfico salvo mostra:
    - Linha SIM: curva obtida por simulação (ground truth)
    - Linha DL: curva predita pelo modelo de deep learning
    """
    # Seleciona a imagem de entrada e a curva verdadeira
    x_img = X[idx : idx + 1]  # Mantém dimensão batch: (1, 128, 128, 1)
    y_true = Y_raw[idx]  # Curva espectral original: (101,)
    y_min = curve_min[idx]  # Parâmetros de desnormalização
    y_max = curve_max[idx]

    # ===================================
    # PIPELINE DE PREDIÇÃO E RECONSTRUÇÃO
    # ===================================
    
    # Passo 1: Predição dos componentes principais PCA
    # O modelo CNN recebe a imagem da geometria e prediz os 30 componentes PCA
    y_pca_pred = model.predict(x_img, verbose=0)[0]  # (30,) - componentes PCA preditos

    # Passo 2: Reconstrução da curva normalizada via transformação inversa do PCA
    # Converte os 30 componentes PCA de volta para 101 pontos espectrais
    y_norm_pred = pca.inverse_transform(y_pca_pred.reshape(1, -1))[0]  # (101,) valores em [0, 1]

    # Passo 3: Desnormalização para a escala original
    # Reverte a normalização min-max aplicada durante o pré-processamento
    # Fórmula: Y_original = Y_norm * (max - min) + min
    y_pred = y_norm_pred * (y_max - y_min) + y_min  # (101,) em escala original  # (101,) em escala original

    # Extrai metadados da amostra para nomear o arquivo
    struct_id = int(metadata[idx, 0])  # ID da estrutura
    angle = int(metadata[idx, 1])  # Ângulo de rotação em graus

    # ===================================
    # PLOTAGEM: SIM vs DL
    # ===================================
    plt.figure(figsize=(6, 4))
    
    # Plota curva simulada (ground truth)
    plt.plot(wavelengths, y_true, label="SIM", linewidth=2)
    
    # Plota curva predita pelo modelo
    plt.plot(wavelengths, y_pred, label="DL", linewidth=2, linestyle="--")
    
    # Configurações do gráfico
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption (a.u.)")
    plt.title(f"Absorption - Estrutura {struct_id:02d}, {angle}°")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Salva figura com nome identificando estrutura e ângulo
    out_path = COMP_DIR / f"comparison_{struct_id:02d}_{angle}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figura salva: {out_path}")


# =========================
# EXECUÇÃO PRINCIPAL
# =========================
if __name__ == "__main__":
    """
    Gera gráficos de comparação para exemplos selecionados.
    
    Escolhe alguns índices representativos do dataset e gera
    visualizações comparativas entre curvas simuladas (SIM) e
    preditas pelo modelo (DL).
    
    Os gráficos gerados permitem avaliar visualmente:
    - Quão bem o modelo reproduz a forma das curvas
    - Se o modelo captura picos e vales importantes
    - Possíveis regiões espectrais com maior/menor erro
    """
    # Replica a mesma divisão do train_model.py (random_state=42)
    # para identificar apenas as amostras do conjunto de teste
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    _, idx_temp = train_test_split(indices, test_size=0.3, random_state=42)
    _, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

    print(f"\n=== Gerando comparações SIM vs DL (conjunto de teste: {len(idx_test)} amostras) ===")
    for idx in sorted(idx_test):
        struct_id = int(metadata[idx, 0])
        angle     = int(metadata[idx, 1])
        print(f"  Estrutura {struct_id:02d}, {angle}°")
        plot_example(idx)
    
    print(f"\n✓ Gráficos salvos em: {COMP_DIR.resolve()}")