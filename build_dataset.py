"""
Script de Preparação do Dataset para Treinamento

Este script processa os dados brutos de geometrias e espectros de absorção,
preparando-os para o treinamento do modelo de deep learning. O processo inclui:

1. Carregamento de imagens de geometrias de nanoestruturas
2. Carregamento de curvas espectrais de absorção
3. Normalização das curvas (min-max scaling)
4. Redução de dimensionalidade via PCA (Principal Component Analysis)
5. Salvamento dos dados processados em formato otimizado (.npy)

O objetivo é transformar imagens de geometrias (entrada) e curvas espectrais (saída)
em um formato adequado para treinar uma CNN que prediz espectros a partir de geometrias.
"""

from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.decomposition import PCA
import joblib

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
# Diretório raiz do projeto
BASE_DIR = Path(__file__).parent

# Diretórios de entrada com dados brutos
DATASET_INPUT_DIR = BASE_DIR / "dataset" / "input"  # Arquivos de geometria (.txt)
DATASET_OUTPUT_DIR = BASE_DIR / "dataset" / "output"  # Arquivos de espectro (.txt)

# Diretórios de resultados
GEOM_DIR = BASE_DIR / "results" / "geometries"  # Imagens de geometria geradas
PREP_DIR = BASE_DIR / "results" / "prepared"  # Dados processados para treinamento
PREP_DIR.mkdir(parents=True, exist_ok=True)

# Estruturas presentes no dataset (estrutura 06 não existe nos dados)
STRUCTURES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]

# Ângulos de rotação disponíveis para cada geometria (em graus)
ANGLES = [0, 15, 30, 45, 60, 75]

# Parâmetros de processamento
IMG_SIZE = 128  # Resolução das imagens de entrada para a CNN (128x128 pixels)
N_PCA = 30  # Número de componentes principais para reduzir dimensionalidade das curvas (101 -> 30)


def load_spectrum(struct_id: int, angle_deg: float):
    """
    Carrega a curva espectral de absorção para uma estrutura e ângulo específicos.
    
    Args:
        struct_id: ID da estrutura (ex: 1, 2, 3, ...)
        angle_deg: Ângulo de rotação em graus (0, 15, 30, 45, 60, 75)
    
    Returns:
        tuple: (wavelengths, field)
            - wavelengths: array com comprimentos de onda (nm)
            - field: array com valores de campo elétrico/absorção
    
    Formato do arquivo:
        - Coluna 0: ângulo de rotação
        - Coluna 1: comprimento de onda (lambda)
        - Coluna 2: campo elétrico/absorção
        - Linhas iniciando com '%' são comentários (ignorados)
    """
    out_path = DATASET_OUTPUT_DIR / f"Output_{struct_id:02d}.txt"

    # Carrega dados ignorando linhas de comentário (iniciadas com '%')
    data = np.loadtxt(out_path, comments="%", dtype=float)

    # Garante que o array seja 2D mesmo se houver apenas uma linha válida
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Filtra apenas as linhas correspondentes ao ângulo desejado
    subset = data[data[:, 0] == angle_deg]
    lambdas = subset[:, 1]  # Comprimentos de onda
    field = subset[:, 2]  # Valores de absorção
    return lambdas, field


def load_geometry_image(struct_id: int, angle_deg: float):
    """
    Carrega e processa uma imagem de geometria de nanoestrutura.
    
    Args:
        struct_id: ID da estrutura (ex: 1, 2, 3, ...)
        angle_deg: Ângulo de rotação em graus (0, 15, 30, 45, 60, 75)
    
    Returns:
        np.ndarray: Array de forma (128, 128, 1) com valores normalizados em [0, 1]
                    Pronto para ser usado como entrada da CNN
    
    Processamento aplicado:
        1. Carrega imagem PNG
        2. Converte para escala de cinza (se for RGB/RGBA)
        3. Redimensiona para 128x128 pixels
        4. Normaliza valores para intervalo [0, 1]
        5. Adiciona dimensão de canal (formato esperado pela CNN)
    """
    img_path = GEOM_DIR / f"geom_{struct_id:02d}_{angle_deg}.png"
    img = imread(img_path)

    # Converte para escala de cinza se a imagem tiver múltiplos canais (RGB/RGBA)
    # Pega apenas o primeiro canal, pois a geometria é binária (preto/branco)
    if img.ndim == 3:
        img = img[..., 0]

    # Redimensiona para o tamanho padrão (128x128) com anti-aliasing para suavizar
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img = img.astype(np.float32)

    # Normaliza os valores de pixel para o intervalo [0, 1]
    max_val = img.max()
    if max_val > 0:
        img = img / max_val

    # Adiciona dimensão de canal: (H, W) -> (H, W, 1)
    # Formato necessário para entrada em redes convolucionais
    img = img[..., np.newaxis]
    return img


def main():
    """
    Função principal que executa todo o pipeline de preparação dos dados.
    
    Processo completo:
    1. Carrega todas as imagens de geometrias e curvas espectrais
    2. Normaliza as curvas usando min-max scaling individual
    3. Aplica PCA para reduzir dimensionalidade das curvas (101 -> 30 componentes)
    4. Salva todos os dados processados em formato .npy
    
    Dados gerados:
    - X_images.npy: Imagens de geometria (entrada do modelo)
    - Y_raw.npy: Curvas espectrais originais (referência)
    - Y_pca.npy: Componentes PCA das curvas (saída do modelo)
    - wavelengths.npy: Comprimentos de onda
    - metadata.npy: Informações de estrutura e ângulo
    - curve_min.npy, curve_max.npy: Parâmetros para desnormalização
    - pca_model.pkl: Modelo PCA treinado (para transformação inversa)
    """
    # Listas para acumular dados durante o carregamento
    X_list = []  # Imagens de geometria
    Y_list = []  # Curvas espectrais
    meta_list = []  # Metadados [struct_id, ângulo]
    y_min_list = []  # Valores mínimos de cada curva (para normalização)
    y_max_list = []  # Valores máximos de cada curva (para normalização)
    wavelengths = None  # Comprimentos de onda (assumidos iguais para todas as estruturas)

    # ==============================
    # 1. Carregar imagens + curvas
    # ==============================
    # Itera sobre todas as combinações de estrutura x ângulo
    # Total: 17 estruturas × 6 ângulos = 102 amostras
    for struct_id in STRUCTURES:
        for ang in ANGLES:
            # Carrega imagem da geometria (entrada X)
            img = load_geometry_image(struct_id, ang)
            X_list.append(img)

            # Carrega curva espectral de absorção (saída Y)
            lambdas, field = load_spectrum(struct_id, ang)

            # Salva os comprimentos de onda na primeira iteração
            # (assumindo que são os mesmos para todas as estruturas)
            if wavelengths is None:
                wavelengths = lambdas

            Y_list.append(field)
            meta_list.append([struct_id, ang])  # Guarda metadados para referência

            # Armazena min/max de cada curva para normalização posterior
            y_min_list.append(field.min())
            y_max_list.append(field.max())

            print(f"Carregado: estrutura {struct_id:02d}, {ang}°")

    # Converte listas para arrays numpy para processamento eficiente
    # N = número total de amostras (17 estruturas × 6 ângulos = 102)
    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 128, 128, 1) - imagens de entrada
    Y_raw = np.stack(Y_list, axis=0).astype(np.float32)  # (N, 101) - curvas espectrais originais
    metadata = np.array(meta_list, dtype=np.float32)  # (N, 2) - [struct_id, ângulo]
    curve_min = np.array(y_min_list, dtype=np.float32)  # (N,) - mínimo de cada curva
    curve_max = np.array(y_max_list, dtype=np.float32)  # (N,) - máximo de cada curva
    wavelengths = np.array(wavelengths, dtype=np.float32)  # (101,) - comprimentos de onda

    print("\nShapes:")
    print("X:", X.shape)
    print("Y_raw:", Y_raw.shape)
    print("metadata:", metadata.shape)

    # =========================================
    # 2. Normalizar curvas (min-max por curva)
    # =========================================
    # Normalização min-max INDIVIDUAL para cada curva
    # Fórmula: Y_norm = (Y - min) / (max - min)
    # Isso transforma cada curva para o intervalo [0, 1]
    # 
    # Por que normalizar individualmente?
    # - Cada geometria tem intensidade de absorção diferente
    # - Queremos que o modelo aprenda a FORMA da curva, não a amplitude absoluta
    # - A amplitude pode ser recuperada posteriormente na desnormalização
    
    denom = curve_max - curve_min  # Amplitude de cada curva
    # Evita divisão por zero para curvas completamente planas (se existirem)
    denom[denom == 0] = 1.0

    # Broadcasting: curve_min[:, None] transforma (N,) em (N, 1) para subtrair de (N, 101)
    Y_norm = (Y_raw - curve_min[:, None]) / denom[:, None]  # (N, 101) valores em [0, 1]  # (N, 101) valores em [0, 1]

    # ==========================
    # 3. PCA nas curvas normalizadas
    # ==========================
    # Aplicação de PCA (Principal Component Analysis) para redução de dimensionalidade
    # 
    # Por que usar PCA?
    # - Curvas espectrais têm 101 pontos (alta dimensionalidade)
    # - Muitos desses pontos são correlacionados
    # - PCA identifica padrões principais e reduz para 30 componentes
    # - Facilita o treinamento da rede neural (menos parâmetros na saída)
    # - Mantém ~95-99% da informação original com apenas 30 componentes
    # 
    # Processo:
    # - Entrada: Y_norm (N, 101) - curvas normalizadas
    # - Saída: Y_pca (N, 30) - componentes principais
    # - O modelo PCA é salvo para permitir transformação inversa depois
    
    print("\nCalculando PCA nas curvas normalizadas...")
    pca = PCA(n_components=N_PCA)  # Inicializa PCA para 30 componentes
    Y_pca = pca.fit_transform(Y_norm)  # Treina PCA e transforma os dados

    # Calcula e exibe quanto da variância original cada componente captura
    explained = pca.explained_variance_ratio_ * 100.0
    print("Y_pca shape:", Y_pca.shape)
    print("Variância explicada (%):", np.round(explained, 3))
    print("Variância total explicada: {:.3f} %".format(explained.sum()))

    # ==========================
    # 4. Salvar tudo
    # ==========================
    # Salva todos os dados processados em formato binário numpy (.npy)
    # Formato .npy é eficiente e preserva tipos de dados
    # 
    # Arquivos salvos:
    
    # Dados de entrada/saída para treinamento:
    np.save(PREP_DIR / "X_images.npy", X)  # Imagens de geometria (entrada da CNN)
    np.save(PREP_DIR / "Y_raw.npy", Y_raw)  # Curvas originais (para avaliação final)
    np.save(PREP_DIR / "Y_pca.npy", Y_pca)  # Componentes PCA (saída da CNN)
    
    # Metadados e parâmetros:
    np.save(PREP_DIR / "wavelengths.npy", wavelengths)  # Eixo x das curvas
    np.save(PREP_DIR / "metadata.npy", metadata)  # Info de estrutura e ângulo
    
    # Parâmetros de normalização (necessários para desnormalização):
    np.save(PREP_DIR / "curve_min.npy", curve_min)  # Mínimo de cada curva
    np.save(PREP_DIR / "curve_max.npy", curve_max)  # Máximo de cada curva

    # Salva o modelo PCA treinado usando joblib
    # Necessário para reconstruir curvas completas a partir dos componentes preditos
    joblib.dump(pca, PREP_DIR / "pca_model.pkl")

    print("\nArquivos salvos em:", PREP_DIR.resolve())


# =========================
# EXECUÇÃO PRINCIPAL
# =========================
if __name__ == "__main__":
    """
    Executa o pipeline completo de preparação de dados:
    1. Carrega imagens de geometrias e curvas espectrais
    2. Normaliza e aplica PCA nas curvas
    3. Salva dados processados em results/prepared/
    
    Execute este script ANTES do treinamento do modelo.
    """
    main()