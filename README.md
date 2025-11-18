# Projeto: Predição de Curvas de Absorção a partir de Geometrias 2D

## Objetivo

Desenvolver um modelo de deep learning capaz de **prever curvas de absorção (espectros)** geradas por simulação (COMSOL) **a partir da geometria 2D** de nanoestruturas, substituindo simulações custosas por um modelo rápido baseado em imagens.

---

# Pipeline Completo

## 1. Entrada dos Dados

### 1.1 Geometrias

- Arquivos `Imput_XX.txt` contendo coordenadas de um polígono.
- Representam a geometria da nanoestrutura.

### 1.2 Espectros

- Arquivos `Output_XX.txt` contendo:
  - ângulo de incidência
  - comprimento de onda
  - valor do campo/absorção
- Para cada estrutura existem simulações nos ângulos:
  `0°, 15°, 30°, 45°, 60°, 75°`.

---

## 2. Geração das Imagens das Geometrias (`geometries.py`)

Para cada estrutura e ângulo:

1. Lê coordenadas da geometria.
2. Centraliza a estrutura no centróide.
3. Rotaciona a geometria no ângulo especificado.
4. Normaliza o tamanho com escala fixa.
5. Renderiza uma imagem 128×128 px (preto no branco).
6. Salva em `results/geometries/geom_ID_angulo.png`.

Total: **90 imagens** de 128×128×1.

---

## 3. Leitura e Organização dos Espectros

Para cada estrutura e ângulo:

1. Lê o arquivo `Output_XX.txt`.
2. Ignora cabeçalhos com `%`.
3. Seleciona apenas o ângulo desejado.
4. Ordena pela coluna do comprimento de onda.
5. Extrai a curva com 101 pontos.

As curvas são salvas posteriormente em `Y_raw.npy`.

---

## 4. Construção do Dataset (`build_dataset.py`)

O script gera:

### 4.1 Imagens (X)

(90, 128, 128, 1)

### 4.2 Curvas originais (Y_raw)

(90, 101)

### 4.3 Metadados

(90, 2) → [estrutura, ângulo]

### 4.4 Normalização das curvas

Cada curva é normalizada individualmente com min–max:

\[
y*{norm} = \frac{y - y*{min}}{y*{max} - y*{min}}
\]

Salvo em:

- `curve_min.npy`
- `curve_max.npy`

---

## 5. PCA nas Curvas Normalizadas

- PCA aplicado em `Y_norm`.
- **20 componentes principais**.
- Variância explicada total: **≈ 99.99%**.

Arquivos salvos:

- `Y_pca.npy` → shape (90, 20)
- `pca_model.pkl`

---

## 6. Modelo de Deep Learning (`train_model.py`)

### Arquitetura

Conv2D(8) + MaxPool
Conv2D(16) + MaxPool
Conv2D(32) + MaxPool
Flatten
Dense(64, relu)
Dropout(0.3)
Dense(20, linear)

### Hiperparâmetros

| Parâmetro     | Valor                            |
| ------------- | -------------------------------- |
| Otimizador    | Adam                             |
| Loss          | MSE                              |
| Métrica       | MAE                              |
| Batch size    | 8                                |
| EarlyStopping | patience=30                      |
| Split         | 70% treino / 15% val / 15% teste |

O modelo aprende a prever os **20 componentes do PCA** a partir da imagem.

Modelo salvo em:
results/models/cnn_pca_model.keras

---

## 7. Reconstrução das Curvas e Comparação (`compare_curves.py`)

Passos para cada exemplo:

1. Predição dos 20 componentes PCA.
2. Aplicação do PCA inverso.
3. Desnormalização (usando min–max da curva original).
4. Comparação visual SIM (simulado) vs DL (modelo).

Saída salva em:
results/comparisons/comparison_ID_angulo.png

---

# Resultado Geral

- O modelo reconstrói com alta precisão as curvas simuladas.
- Picos e forma geral coincidem muito bem.
- A normalização + PCA=20 foi crucial para estabilizar a aprendizagem.
- Para apenas 90 exemplos, o desempenho é excelente.

---

# Parâmetros Principais (Resumo)

## Dados

- 15 estruturas (exceto 06)
- 6 ângulos por estrutura
- Total: 90 amostras

## Imagens

- 128×128 px
- 1 canal

## Curvas

- 101 pontos

## PCA

- 20 componentes
- Variância explicada ~99.99%

## CNN

- 3 convoluções + 3 maxpools
- Dense(64) + Dropout(0.3)
- Saída: 20 valores (PCA)

## Treinamento

- 70/15/15 split
- EarlyStopping (30)
- MSE + MAE

---

# Conclusão

O pipeline implementa:

1. Geração de geometrias rotacionadas.
2. Processamento dos espectros simulados.
3. Normalização e PCA.
4. Treinamento de uma CNN para prever componentes principais.
5. Reconstrução das curvas e comparação com o COMSOL.

O sistema é capaz de prever curvas de absorção de forma rápida e com alta precisão.
