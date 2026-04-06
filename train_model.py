"""
Script de Treinamento do Modelo CNN+PCA

Este script treina uma rede neural convolucional (CNN) para predizer componentes PCA
de curvas espectrais a partir de imagens de geometrias de nanoestruturas.

Melhorias anti-overfitting aplicadas:
1. Data augmentation (flip + ruído gaussiano, apenas no treino)
2. Modelo menor (8->16->32 filtros, Dense 64->32)
3. Dropout aumentado (0.5 e 0.3)
4. Regularização L2 nas camadas Dense
5. Avaliação por k-fold cross-validation ao final
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# =========================
# CARREGAR DADOS PREPARADOS
# =========================
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"
MODELS_DIR = BASE_DIR / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(PREP_DIR / "X_images.npy")   # (N,128,128,1) - já normalizado [0,1]
Y = np.load(PREP_DIR / "Y_pca.npy")       # (N,30) - 30 componentes principais

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# DIVISÃO TREINO / VAL / TESTE
# =========================
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

print("\nDivisão dos dados:")
print("Treino   :", X_train.shape[0])
print("Validação:", X_val.shape[0])
print("Teste    :", X_test.shape[0])

# =========================
# FUNÇÃO DE CONSTRUÇÃO DO MODELO
# =========================
def build_model(input_shape=(128, 128, 1), n_outputs=30):
    """
    CNN menor + MLP com regularização.

    Mudanças vs versão anterior:
    - Filtros: 16->32->64  =>  8->16->32  (menos parâmetros)
    - Dense:   128->64     =>  64->32     (menos parâmetros)
    - Dropout: 0.35, 0.2   =>  0.50, 0.30 (mais regularização)
    - L2 regularização nas camadas Dense
    - Data augmentation aplicada como primeiras camadas (só ativa no treino)
    """
    inputs = keras.Input(shape=input_shape)

    # --- Data Augmentation (ativa apenas durante model.fit) ---
    # RandomFlip assume que a resposta espectral é simétrica sob reflexão.
    # Se as geometrias forem assimétricas, remova o RandomFlip.
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.GaussianNoise(0.05)(x)   # ruído aumentado para geometrias binárias

    # --- Blocos Convolucionais (com L2 nas Conv) ---
    x = layers.Conv2D(8,  (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # --- Cabeça MLP (capacidade reduzida + L2) ---
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(n_outputs, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_pca_model")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# =========================
# TREINAMENTO PRINCIPAL
# =========================
input_shape = (128, 128, 1)
n_outputs = Y.shape[1]

model = build_model(input_shape, n_outputs)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=60,
    restore_best_weights=True,
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    min_lr=1e-5,
    verbose=1,
)

history = model.fit(
    X_train, Y_train,
    epochs=400,
    batch_size=16,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)

# =========================
# AVALIAÇÃO EM TESTE
# =========================
print("\n=== Desempenho no conjunto de TESTE ===")
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
train_loss, train_mae = model.evaluate(X_train, Y_train, verbose=0)
val_loss_final, val_mae_final = model.evaluate(X_val, Y_val, verbose=0)

print(f"Train MSE={train_loss:.6f}  MAE={train_mae:.6f}")
print(f"Val   MSE={val_loss_final:.6f}  MAE={val_mae_final:.6f}")
print(f"Test  MSE={test_loss:.6f}  MAE={test_mae:.6f}")
print(f"Val/Train ratio: {val_loss_final/train_loss:.2f}x  (era ~13x antes)")
print(f"Test/Train ratio: {test_loss/train_loss:.2f}x  (era ~16x antes)")

# =========================
# SALVAR MODELO E HISTÓRICO
# =========================
model_path = MODELS_DIR / "cnn_pca_model.keras"
hist_path  = MODELS_DIR / "history.npy"

model.save(model_path)
np.save(hist_path, history.history)

print("\nModelo salvo em:", model_path.resolve())
print("Histórico salvo em:", hist_path.resolve())

# =========================
# 5. K-FOLD CROSS-VALIDATION
# =========================
# Avaliação robusta de generalização usando todos os 102 samples.
# Complementa o treino principal — não substitui o modelo salvo acima.
print("\n=== K-Fold Cross-Validation (k=5) ===")
print("(Avalia generalização real; não altera o modelo salvo)")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_test_losses = []
fold_test_maes   = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    # Valida dentro do fold (10% do treino do fold)
    split = max(1, int(len(X_tr) * 0.1))
    X_tr_f, X_val_f = X_tr[:-split], X_tr[-split:]
    Y_tr_f, Y_val_f = Y_tr[:-split], Y_tr[-split:]

    m = build_model(input_shape, n_outputs)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=40, restore_best_weights=True
    )
    rlr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=0
    )
    m.fit(
        X_tr_f, Y_tr_f,
        epochs=400,
        batch_size=16,
        validation_data=(X_val_f, Y_val_f),
        callbacks=[es, rlr],
        verbose=0,
    )
    loss, mae = m.evaluate(X_te, Y_te, verbose=0)
    fold_test_losses.append(loss)
    fold_test_maes.append(mae)
    print(f"  Fold {fold+1}: test_loss={loss:.4f}  test_mae={mae:.4f}")

print(f"\nCV MSE : {np.mean(fold_test_losses):.4f} ± {np.std(fold_test_losses):.4f}")
print(f"CV MAE : {np.mean(fold_test_maes):.4f} ± {np.std(fold_test_maes):.4f}")
print("\nPróximos passos:")
print("1. Execute analyze_results.py para visualizar histórico de treino")
print("2. Execute compare_curves.py para comparar predições com ground truth")
