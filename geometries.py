import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===========================================
# CONFIGURAÇÃO DE DIRETÓRIOS
# ===========================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = DATASET_DIR / "input"
OUTPUT_DIR = DATASET_DIR / "output"

RESULTS_DIR = BASE_DIR / "results"

GEOM_DIR = RESULTS_DIR / "geometries"

SPEC_DIR = RESULTS_DIR / "spectrum"  # onde salvar espectros

# Criar pastas se não existirem
GEOM_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

ANGLES = [0, 15, 30, 45, 60, 75]
TARGET_SIZE = (128, 128)  # tamanho final da imagem para a CNN


# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================


def rotate_points(x, y, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = R @ np.vstack((x, y))
    return rotated[0], rotated[1]


def load_geometry(path):
    coords = np.loadtxt(path)
    x, y = coords[:, 0], coords[:, 1]
    # fechar polígono
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return x, y


def load_output(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            angle, lam, field = line.split()
            data.append((float(angle), float(lam), float(field)))
    return np.array(data)


# ===========================================
# PROCESSAMENTO DOS INPUTS
# ===========================================

for input_file in sorted(INPUT_DIR.glob("Imput_*.txt")):
    base_name = input_file.stem  # ex: "Imput_01"
    idx = base_name.split("_")[1]  # ex: "01"

    print(f"\n=== Processando estrutura {idx} ===")

    # =======================
    # 1. GERAR GEOMETRIAS
    # =======================

    x, y = load_geometry(input_file)

    for ang in ANGLES:
        xr, yr = rotate_points(x, y, ang)

        # figura temporária
        plt.figure(figsize=(3, 3))

        # fundo branco
        fig = plt.gcf()
        fig.patch.set_facecolor("white")
        ax = plt.gca()
        ax.set_facecolor("white")

        # polígono preto
        plt.fill(xr, yr, color="black", edgecolor="black")

        plt.axis("equal")
        plt.axis("off")

        # primeiro salva uma imagem "bruta" temporária
        raw_path = GEOM_DIR / f"geom_{idx}_{ang}_raw.png"
        plt.tight_layout(pad=0)
        plt.savefig(raw_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # agora padroniza com Pillow: grayscale, 128x128, binarização
        img = Image.open(raw_path).convert("L")  # grayscale
        img = img.resize(TARGET_SIZE)

        # binarizar: geometria preta (0), fundo branco (255)
        img = img.point(lambda p: 0 if p < 200 else 255)

        final_path = GEOM_DIR / f"geom_{idx}_{ang}.png"
        img.save(final_path)

        # opcional: remover o arquivo bruto
        raw_path.unlink(missing_ok=True)

        print(f"   → Geometria salva em {final_path}")

    # =======================
    # 2. GERAR ESPECTROS
    # =======================

    out_path = OUTPUT_DIR / f"Output_{idx}.txt"
    if not out_path.exists():
        print(f"   [AVISO] Output_{idx}.txt não encontrado, pulando.")
        continue

    data = load_output(out_path)
    if data.size == 0:
        print(f"   [AVISO] Output_{idx}.txt vazio, pulando.")
        continue

    angles_output = sorted(list(set(data[:, 0])))

    # --- gráfico único com todas curvas ---

    plt.figure(figsize=(6, 4))
    for ang in angles_output:
        subset = data[data[:, 0] == ang]
        lambdas = subset[:, 1]
        field = subset[:, 2]

        # ordenar por lambda
        order = np.argsort(lambdas)
        lambdas = lambdas[order]
        field = field[order]

        plt.plot(lambdas, field, label=f"{int(ang)}°")

    plt.title(f"Curvas de absorção – Estrutura {idx}")
    plt.xlabel("Comprimento de onda (nm)")
    plt.ylabel("Campo elétrico (unid. sim.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    combined_path = SPEC_DIR / f"spectrum_{idx}_all.png"
    plt.savefig(combined_path, dpi=300)
    plt.close()

    print(f"   → Espectros combinados salvos em {combined_path}")

    # --- espectro por ângulo ---
    for ang in angles_output:
        subset = data[data[:, 0] == ang]
        lambdas = subset[:, 1]
        field = subset[:, 2]

        order = np.argsort(lambdas)
        lambdas = lambdas[order]
        field = field[order]

        plt.figure(figsize=(6, 4))
        plt.plot(lambdas, field)
        plt.title(f"Curva – Estrutura {idx}, {int(ang)}°")
        plt.xlabel("Comprimento de onda (nm)")
        plt.ylabel("Campo elétrico (unid. sim.)")
        plt.grid(True)
        plt.tight_layout()

        spec_path = SPEC_DIR / f"spectrum_{idx}_{ang}.png"
        plt.savefig(spec_path, dpi=300)
        plt.close()

        print(f"   → Espectro salvo em {spec_path}")

print("\n✅ Dataset completo gerado em: results/geometries/imgs e results/spectrum")
