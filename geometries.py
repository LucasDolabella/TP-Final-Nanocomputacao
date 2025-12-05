"""
Script de Geração de Geometrias e Espectros

Este script processa os dados brutos de entrada/saída e gera:
1. Imagens de geometrias de nanoestruturas em diferentes ângulos de rotação
2. Gráficos de espectros de absorção para cada estrutura e ângulo

Processamento de geometrias:
- Lê coordenadas de polígonos dos arquivos Input_XX.txt
- Aplica rotações em múltiplos ângulos (0°, 15°, 30°, 45°, 60°, 75°)
- Gera imagens padronizadas (128x128 pixels, preto/branco)
- Mantém escala fixa para consistência entre estruturas

Processamento de espectros:
- Lê dados espectrais dos arquivos Output_XX.txt
- Gera gráficos de absorção vs comprimento de onda
- Salva visualizações individuais por ângulo

Este é o primeiro passo do pipeline, executado ANTES de build_dataset.py
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===========================================
# CONFIGURAÇÕES DE DIRETÓRIOS
# ===========================================

# Diretório raiz do projeto
BASE_DIR = Path(__file__).parent

# Diretórios de dados brutos (input)
DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = DATASET_DIR / "input"  # Arquivos de geometria (.txt com coordenadas)
OUTPUT_DIR = DATASET_DIR / "output"  # Arquivos de espectro (.txt com ângulo, lambda, campo)

# Diretórios de resultados (output)
RESULTS_DIR = BASE_DIR / "results"
GEOM_DIR = RESULTS_DIR / "geometries"  # Imagens de geometrias geradas
SPEC_DIR = RESULTS_DIR / "spectrum"  # Gráficos de espectros gerados

# Cria diretórios de saída se não existirem
GEOM_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

# Configurações de processamento
ANGLES = [0, 15, 30, 45, 60, 75]  # Ângulos de rotação em graus
TARGET_SIZE = (128, 128)  # Resolução final das imagens (padrão para CNN)


# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================


def rotate_points(x, y, angle_deg):
    """
    Rotaciona pontos 2D em torno da origem usando matriz de rotação.
    
    Args:
        x: array com coordenadas x dos pontos
        y: array com coordenadas y dos pontos
        angle_deg: ângulo de rotação em graus (sentido anti-horário)
    
    Returns:
        tuple: (x_rotacionado, y_rotacionado)
    
    Matriz de rotação 2D:
        R = [[cos(θ), -sin(θ)],
             [sin(θ),  cos(θ)]]
    """
    theta = np.radians(angle_deg)  # Converte graus para radianos
    # Matriz de rotação 2D
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Aplica rotação: R @ [x, y]
    out = R @ np.vstack((x, y))
    return out[0], out[1]


def load_geometry(path):
    """
    Carrega coordenadas de geometria de um arquivo .txt.
    
    Args:
        path: caminho para arquivo Input_XX.txt contendo coordenadas (x, y)
    
    Returns:
        tuple: (x, y) arrays com coordenadas do polígono fechado
    
    Formato do arquivo:
        Cada linha contém: x y
        (coordenadas dos vértices do polígono)
    
    O polígono é fechado automaticamente conectando último ponto ao primeiro.
    """
    coords = np.loadtxt(path)  # Carrega matriz de coordenadas
    x, y = coords[:, 0], coords[:, 1]  # Separa em vetores x e y
    # Fecha o polígono adicionando o primeiro ponto ao final
    return np.append(x, x[0]), np.append(y, y[0])


def load_output(path):
    """
    Carrega dados espectrais de um arquivo Output_XX.txt.
    
    Args:
        path: caminho para arquivo Output_XX.txt
    
    Returns:
        np.ndarray: matriz (N, 3) com [ângulo, comprimento_onda, campo]
    
    Formato do arquivo:
        Cada linha contém: ângulo lambda campo
        Linhas começando com '%' são comentários (ignoradas)
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Ignora linhas vazias e comentários
            if not line or line.startswith("%"):
                continue
            # Parse: ângulo, comprimento de onda, campo elétrico
            angle, lam, field = line.split()
            data.append((float(angle), float(lam), float(field)))
    return np.array(data)


# ===========================================
# PROCESSAMENTO COMPLETO
# ===========================================
# Loop principal: processa cada estrutura (arquivo Input_XX.txt)
# Para cada estrutura, gera:
# - Imagens de geometria em múltiplos ângulos
# - Gráficos de espectros de absorção

for input_file in sorted(INPUT_DIR.glob("Imput_*.txt")):
    # Extrai o ID da estrutura do nome do arquivo
    base_name = input_file.stem  # ex: "Imput_05"
    idx = base_name.split("_")[1]  # ex: "05"

    print(f"\n=== GERANDO ESTRUTURA {idx} ===")

    # ------------------------------
    # 1. GEOMETRIAS
    # ------------------------------
    # Processa geometria: carrega coordenadas, centraliza, rotaciona e gera imagens

    # Carrega coordenadas do polígono
    x, y = load_geometry(input_file)

    # Centraliza o polígono no centróide (centro de massa)
    # Isso garante que rotações sejam em torno do centro da geometria
    cx, cy = np.mean(x), np.mean(y)  # Calcula centróide
    x_c = x - cx  # Translada para origem
    y_c = y - cy

    # Calcula raio máximo com margem de 5%
    # Mantém escala FIXA para todas as estruturas (importante para CNN)
    # Isso garante que geometrias de tamanhos diferentes ocupem espaços comparáveis
    r = np.sqrt(x_c**2 + y_c**2).max() * 1.05  # Raio + 5% de margem

    # Gera imagem para cada ângulo de rotação
    for ang in ANGLES:
        # Aplica rotação nas coordenadas centralizadas
        xr, yr = rotate_points(x_c, y_c, ang)

        # Cria figura matplotlib para renderizar a geometria
        fig, ax = plt.subplots(figsize=(3, 3))

        # Configura fundo branco (importante: geometria preta em fundo branco)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Desenha polígono preenchido em preto
        ax.fill(xr, yr, color="black", edgecolor="black")

        # Configurações de enquadramento FIXO
        # Usa o mesmo raio r para todas as estruturas/ângulos
        # Isso garante escala consistente entre diferentes geometrias
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")  # Mantém aspecto quadrado
        ax.axis("off")  # Remove eixos e bordas

        # Salva imagem bruta em alta resolução
        raw_path = GEOM_DIR / f"geom_{idx}_{ang}_raw.png"
        fig.savefig(raw_path, dpi=300, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        # Pós-processamento da imagem usando PIL
        img = Image.open(raw_path).convert("L")  # Converte para escala de cinza
        img = img.resize(TARGET_SIZE, Image.LANCZOS)  # Redimensiona para 128x128 com anti-aliasing
        
        # Binariza: pixels claros (>200) -> branco (255), escuros (<200) -> preto (0)
        # Garante imagem puramente binária (sem tons de cinza)
        img = img.point(lambda p: 0 if p < 200 else 255)

        # Salva imagem final processada
        final_path = GEOM_DIR / f"geom_{idx}_{ang}.png"
        img.save(final_path)
        
        # Remove arquivo bruto temporário
        raw_path.unlink(missing_ok=True)

        print(f"  ✔ Geometria salva: {final_path}")

    # ------------------------------
    # 2. ESPECTROS
    # ------------------------------
    # Processa dados espectrais: carrega, organiza e plota curvas de absorção

    # Verifica se arquivo de output existe
    output_path = OUTPUT_DIR / f"Output_{idx}.txt"
    if not output_path.exists():
        print(f"  ⚠ Output_{idx}.txt não encontrado!")
        continue

    # Carrega dados espectrais (ângulo, lambda, campo)
    data = load_output(output_path)
    if data.size == 0:
        print(f"  ⚠ Output_{idx}.txt vazio!")
        continue

    # Extrai lista de ângulos disponíveis nos dados
    angles_output = sorted(list(set(data[:, 0])))

    # Gera gráfico de espectro para cada ângulo
    for ang in angles_output:
        # Filtra dados apenas para o ângulo atual
        subset = data[data[:, 0] == ang]
        lambdas = subset[:, 1]  # Comprimentos de onda
        field = subset[:, 2]  # Valores de campo/absorção

        # Ordena por comprimento de onda (para plotagem suave)
        order = np.argsort(lambdas)
        lambdas = lambdas[order]
        field = field[order]

        # Cria gráfico de espectro de absorção
        plt.figure(figsize=(6, 4))
        plt.plot(lambdas, field, linewidth=2)
        plt.title(f"Espectro – Estrutura {idx}, {int(ang)}°")
        plt.xlabel("Comprimento de onda (nm)")
        plt.ylabel("Campo elétrico (unid.)")
        plt.grid(True, alpha=0.3)

        # Salva gráfico de espectro
        spec_file = SPEC_DIR / f"spectrum_{idx}_{int(ang)}.png"
        plt.savefig(spec_file, dpi=300)
        plt.close()

        print(f"  ✔ Espectro salvo: {spec_file}")

# ===========================================
# MENSAGEM FINAL
# ===========================================
print("\n🎉 Dataset final gerado com sucesso!")
print(f"→ Imagens de geometrias: {GEOM_DIR.resolve()}")
print(f"→ Gráficos de espectros: {SPEC_DIR.resolve()}")
print("\nPróximo passo: execute build_dataset.py para preparar dados de treinamento")