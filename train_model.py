"""
Script de Treinamento do Modelo CNN+PCA

Este script treina uma rede neural convolucional (CNN) para predizer componentes PCA
de curvas espectrais a partir de imagens de geometrias de nanoestruturas.

Arquitetura do modelo:
- Entrada: Imagens 128x128 (geometrias de nanoestruturas)
- Extrator de features: CNN com 3 blocos convolucionais
- Saída: 30 componentes principais PCA (representação compacta de curvas espectrais)

Pipeline completo:
1. Carrega dados preparados (X_images, Y_pca)
2. Divide em treino (70%), validação (15%) e teste (15%)
3. Define arquitetura CNN+MLP
4. Treina com early stopping e learning rate scheduling
5. Avalia desempenho no conjunto de teste
6. Salva modelo treinado e histórico

O modelo prediz componentes PCA que podem ser reconstituídos em curvas completas
usando o modelo PCA salvo em build_dataset.py
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CARREGAR DADOS PREPARADOS
# =========================
# Define diretórios do projeto
BASE_DIR = Path(__file__).parent
PREP_DIR = BASE_DIR / "results" / "prepared"  # Dados processados pelo build_dataset.py
MODELS_DIR = BASE_DIR / "results" / "models"  # Destino para modelo treinado
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Carrega dados preparados
# X: imagens de geometrias (entrada da CNN)
# Y: componentes PCA das curvas espectrais (saída da CNN)
X = np.load(PREP_DIR / "X_images.npy")  # (N,128,128,1) - imagens normalizadas
Y = np.load(PREP_DIR / "Y_pca.npy")  # (N,30) - 30 componentes principais

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# DIVISÃO TREINO / VAL / TESTE
# =========================
# Divide o dataset em 3 conjuntos:
# - Treino (70%): usado para atualizar pesos do modelo
# - Validação (15%): monitora overfitting durante treinamento
# - Teste (15%): avaliação final em dados nunca vistos
#
# Divisão em 2 etapas para obter proporção 70-15-15:

# Primeira divisão: 70% treino, 30% temporário
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42  # random_state garante reprodutibilidade
)

# Segunda divisão: divide temporário em validação e teste (50-50)
# Resulta em 15% validação e 15% teste do total
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

print("\nDivisão dos dados:")
print("Treino   :", X_train.shape[0])
print("Validação:", X_val.shape[0])
print("Teste    :", X_test.shape[0])

# =========================
# DEFINIR MODELO CNN + MLP
# =========================
# Arquitetura: CNN (extração de features) + MLP (regressão)
# A CNN extrai características visuais das geometrias
# A MLP transforma essas features em componentes PCA

input_shape = (128, 128, 1)  # Imagens 128x128 pixels, 1 canal (grayscale)
n_outputs = Y.shape[1]  # 30 componentes do PCA

# Define camada de entrada
inputs = keras.Input(shape=input_shape)

# ===================================
# BLOCOS CONVOLUCIONAIS (Feature Extraction)
# ===================================
# Cada bloco: Conv2D -> MaxPooling2D
# Conv2D detecta padrões locais, MaxPooling reduz dimensionalidade

# Bloco 1: 16 filtros 3x3, reduz 128x128 -> 64x64
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2, 2))(x)

# Bloco 2: 32 filtros 3x3, reduz 64x64 -> 32x32
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Bloco 3: 64 filtros 3x3, reduz 32x32 -> 16x16
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Achata tensor 3D em vetor 1D: (16, 16, 64) -> (16384,)
x = layers.Flatten()(x)

# ===================================
# CABEÇA MLP (Regression Head)
# ===================================
# Camadas densas que mapeiam features visuais para componentes PCA
# Dropout previne overfitting

# Camada densa 1: 128 neurônios
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.35)(x)  # Desliga 35% dos neurônios aleatoriamente

# Camada densa 2: 64 neurônios
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)  # Desliga 20% dos neurônios

# Camada de saída: 30 neurônios (1 para cada componente PCA)
# Ativação linear permite valores reais (não apenas [0,1])
outputs = layers.Dense(n_outputs, activation="linear")(x)

# Cria modelo Keras
model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_pca_model")

# Exibe arquitetura completa do modelo
model.summary()

# ===================================
# COMPILAÇÃO DO MODELO
# ===================================
# Define otimizador, função de perda e métricas
model.compile(
    optimizer="adam",  # Adam: otimizador adaptativo eficiente
    loss="mse",  # MSE (Mean Squared Error): adequado para regressão
    metrics=["mae"],  # MAE (Mean Absolute Error): métrica mais interpretável
)

# =========================
# CALLBACKS (EarlyStopping + LR Scheduler)
# =========================
# Callbacks: funções chamadas durante o treinamento para ajustar comportamento

# Early Stopping: Para treinamento se validação não melhorar
# Previne overfitting e economiza tempo de treino
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitora perda no conjunto de validação
    patience=60,  # Espera 60 épocas sem melhora antes de parar
    restore_best_weights=True,  # Restaura pesos da melhor época (não a última)
)

# Learning Rate Scheduler: Reduz taxa de aprendizado quando estagnado
# Ajuda a convergir melhor em mínimos locais
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",  # Monitora perda no conjunto de validação
    factor=0.5,  # Multiplica learning rate por 0.5 (reduz pela metade)
    patience=15,  # Espera 15 épocas sem melhora antes de reduzir
    min_lr=1e-5,  # Learning rate mínimo
    verbose=1,  # Exibe mensagem quando reduz learning rate
)

# =========================
# TREINAMENTO
# =========================
# Executa o loop de treinamento
# O modelo aprende a mapear geometrias -> componentes PCA
history = model.fit(
    X_train,
    Y_train,
    epochs=400,  # Máximo de 400 épocas (early stopping pode parar antes)
    batch_size=16,  # Processa 16 amostras por vez antes de atualizar pesos
    validation_data=(X_val, Y_val),  # Valida a cada época
    callbacks=[early_stop, reduce_lr],  # Aplica early stopping e LR scheduling
    verbose=2,  # Exibe progresso (1 linha por época)
)

# =========================
# AVALIAÇÃO EM TESTE
# =========================
# Avalia desempenho final em dados nunca vistos durante treinamento
# Fornece estimativa realista da capacidade de generalização
print("\n=== Desempenho no conjunto de TESTE ===")
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test MSE (loss): {test_loss:.6f}")
print(f"Test MAE       : {test_mae:.6f}")

# =========================
# SALVAR MODELO E HISTÓRICO
# =========================
# Salva modelo treinado para uso posterior em análises e predições
model_path = MODELS_DIR / "cnn_pca_model.keras"
hist_path = MODELS_DIR / "history.npy"

# Salva modelo completo (arquitetura + pesos)
model.save(model_path)

# Salva histórico de treinamento (loss e métricas por época)
# Útil para plotar curvas de aprendizado
np.save(hist_path, history.history)

print("\nModelo salvo em:", model_path.resolve())
print("Histórico salvo em:", hist_path.resolve())
print("\nPróximos passos:")
print("1. Execute analyze_results.py para visualizar histórico de treino")
print("2. Execute compare_curves.py para comparar predições com ground truth")