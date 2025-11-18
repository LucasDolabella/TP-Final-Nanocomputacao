import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===========================================
# CONFIGURAÇÕES DE DIRETÓRIOS
# ===========================================

BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = DATASET_DIR / "input"
OUTPUT_DIR = DATASET_DIR / "output"

RESULTS_DIR = BASE_DIR / "results"
GEOM_DIR = RESULTS_DIR / "geometries"
SPEC_DIR = RESULTS_DIR / "spectrum"

# Criar pastas
GEOM_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

# Configurações
ANGLES = [0, 15, 30, 45, 60, 75]
TARGET_SIZE = (128, 128)  # tamanho da imagem para CNN


# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================


def rotate_points(x, y, angle_deg):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = R @ np.vstack((x, y))
    return out[0], out[1]


def load_geometry(path):
    coords = np.loadtxt(path)
    x, y = coords[:, 0], coords[:, 1]
    return np.append(x, x[0]), np.append(y, y[0])  # fechar polígono


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
# PROCESSAMENTO COMPLETO
# ===========================================

for input_file in sorted(INPUT_DIR.glob("Imput_*.txt")):
    base_name = input_file.stem  # ex: "Imput_05"
    idx = base_name.split("_")[1]  # "05"

    print(f"\n=== GERANDO ESTRUTURA {idx} ===")

    # ------------------------------
    # 1. GEOMETRIAS
    # ------------------------------

    x, y = load_geometry(input_file)

    # Centralizar no centróide
    cx, cy = np.mean(x), np.mean(y)
    x_c = x - cx
    y_c = y - cy

    # Raio máximo para manter escala FIXA (agora 5% apenas)
    r = np.sqrt(x_c**2 + y_c**2).max() * 1.05

    for ang in ANGLES:
        xr, yr = rotate_points(x_c, y_c, ang)

        fig, ax = plt.subplots(figsize=(3, 3))

        # fundo branco
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # polígono preto
        ax.fill(xr, yr, color="black", edgecolor="black")

        # enquadramento FIXO
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.axis("off")

        # salvar imagem bruta
        raw_path = GEOM_DIR / f"geom_{idx}_{ang}_raw.png"
        fig.savefig(raw_path, dpi=300, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        # abrir imagem e padronizar
        img = Image.open(raw_path).convert("L")  # grayscale
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img = img.point(lambda p: 0 if p < 200 else 255)

        final_path = GEOM_DIR / f"geom_{idx}_{ang}.png"
        img.save(final_path)
        raw_path.unlink(missing_ok=True)

        print(f"  ✔ Geometria salva: {final_path}")

    # ------------------------------
    # 2. ESPECTROS (SEM COMBINADO)
    # ------------------------------

    output_path = OUTPUT_DIR / f"Output_{idx}.txt"
    if not output_path.exists():
        print(f"  ⚠ Output_{idx}.txt não encontrado!")
        continue

    data = load_output(output_path)
    if data.size == 0:
        print(f"  ⚠ Output_{idx}.txt vazio!")
        continue

    angles_output = sorted(list(set(data[:, 0])))

    # --- espectro por ângulo ---
    for ang in angles_output:
        subset = data[data[:, 0] == ang]
        lambdas = subset[:, 1]
        field = subset[:, 2]

        # ordenar por lambda
        order = np.argsort(lambdas)
        lambdas = lambdas[order]
        field = field[order]

        plt.figure(figsize=(6, 4))
        plt.plot(lambdas, field)
        plt.title(f"Espectro – Estrutura {idx}, {int(ang)}°")
        plt.xlabel("Comprimento de onda (nm)")
        plt.ylabel("Campo elétrico (unid.)")
        plt.grid(True)

        spec_file = SPEC_DIR / f"spectrum_{idx}_{int(ang)}.png"
        plt.savefig(spec_file, dpi=300)
        plt.close()

        print(f"  ✔ Espectro salvo: {spec_file}")

print("\n🎉 Dataset final gerado com sucesso!")
print("→ Imagens: results/geometries/imgs")
print("→ Espectros: results/spectrum")
