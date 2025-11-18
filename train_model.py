from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CARREGAR DADOS PREPARADOS
# =========================
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"
MODELS_DIR = BASE_DIR / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(PREP_DIR / "X_images.npy")  # (N,128,128,1)
Y = np.load(PREP_DIR / "Y_pca.npy")  # (N,20)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# DIVISÃO TREINO / VAL / TESTE
# =========================
# 1º: separa 70% treino, 30% temporário
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

# 2º: divide o temporário em 50% val, 50% teste  -> 15% / 15% do total
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

print("\nDivisão dos dados:")
print("Treino :", X_train.shape[0])
print("Validação:", X_val.shape[0])
print("Teste  :", X_test.shape[0])

# =========================
# DEFINIR MODELO CNN + MLP
# =========================
input_shape = (128, 128, 1)
n_outputs = Y.shape[1]  # 20 componentes do PCA

inputs = keras.Input(shape=input_shape)

x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(n_outputs, activation="linear")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_pca_model")

model.summary()

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

# =========================
# TREINAMENTO (com EarlyStopping)
# =========================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=38,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    Y_train,
    epochs=300,
    batch_size=8,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop],
    verbose=2,
)

# =========================
# AVALIAÇÃO EM TESTE
# =========================
print("\n=== Desempenho no conjunto de TESTE ===")
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MSE (loss): {test_loss:.6f}")
print(f"Test MAE      : {test_mae:.6f}")

# =========================
# SALVAR MODELO E HISTÓRICO
# =========================
model_path = MODELS_DIR / "cnn_pca_model.keras"
hist_path = MODELS_DIR / "history.npy"

model.save(model_path)
np.save(hist_path, history.history)

print("\nModelo salvo em:", model_path.resolve())
print("Histórico salvo em:", hist_path.resolve())
