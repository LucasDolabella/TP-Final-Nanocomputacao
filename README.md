# Projeto: Predição de Curvas de Absorção a partir de Geometrias 2D

## Objetivo

Desenvolver um modelo de deep learning capaz de **prever curvas de absorção (espectros)** geradas via simulação (COMSOL) **a partir da geometria 2D** de nanoestruturas.  
A meta é substituir simulações custosas por um **modelo rápido, robusto e acurado**, baseado em imagens.

---

# Pipeline Completo

## 1. Entrada dos Dados

### 1.1 Geometrias (dataset/input)

Arquivos `Imput_XX.txt`, cada um contendo coordenadas de um polígono.  
Representam a geometria 2D da nanoestrutura.

### 1.2 Espectros (dataset/output)

Arquivos `Output_XX.txt`, contendo:

- Ângulo de incidência
- Comprimento de onda (λ)
- Valor do campo / absorção

Cada estrutura possui simulações em **6 ângulos**:
0°, 15°, 30°, 45°, 60°, 75°

---

## 2. Geração das Imagens das Geometrias (`geometries.py`)

Para cada estrutura e ângulo:

1. Lê o arquivo `Imput_XX.txt`
2. Centraliza o polígono
3. Rotaciona para o ângulo desejado
4. Mantém escala fixa (raio máximo × 1.05)
5. Renderiza imagem 128×128 px em escala de cinza
6. Faz binarização (0/255)
7. Salva em:  
   `results/geometries/geom_ID_angulo.png`

**Total:**  
**126 imagens** (21 estruturas × 6 ângulos).

---

## 3. Leitura e Organização dos Espectros

Para cada estrutura:

- Lê `Output_XX.txt`
- Ignora cabeçalhos iniciados com `%`
- Separa apenas as linhas do ângulo desejado
- Ordena pelos comprimentos de onda
- Salva as curvas com 101 pontos

As curvas originais são armazenadas em:
results/prepared/Y_raw.npy

---

# 4. Construção do Dataset (`build_dataset.py`)

O script gera:

### 4.1 Imagens (X)

X_images.npy → shape (126, 128, 128, 1)

### 4.2 Curvas Originais (Y_raw)

Y_raw.npy → shape (126, 101)

### 4.3 Metadados

metadata.npy → (estrutura, ângulo)

### 4.4 Normalização das curvas (min–max por curva)

\[
y*{\text{norm}} = \frac{y - y*{\min}}{y*{\max} - y*{\min}}
\]

Valores armazenados:

- `curve_min.npy`
- `curve_max.npy`

---

# 5. PCA nas Curvas Normalizadas

Aplicado sobre `Y_norm`.

### Parâmetros:

- **N_PCA = 30 componentes**
- Variância explicada ≈ **100%**

Arquivos gerados:

- `Y_pca.npy` → shape (126, 30)
- `pca_model.pkl`

---

# 6. Modelo de Deep Learning (`train_model.py`)

Rede composta por uma **CNN (extrator de features)** + **MLP (regressor PCA)**, com diversas melhorias anti-overfitting aplicadas em relação à versão inicial.

## Arquitetura Final

### Data Augmentation (apenas durante treino)

RandomFlip (horizontal e vertical)
GaussianNoise (σ = 0.02)

### CNN (capacidade reduzida)

Conv2D(8) + MaxPool
Conv2D(16) + MaxPool
Conv2D(32) + MaxPool
Flatten

### MLP

Dense(64, relu) + L2(1e-4)
Dropout(0.5)
Dense(32, relu) + L2(1e-4)
Dropout(0.3)
Dense(30, linear) # saída = 30 componentes PCA

## Melhorias Anti-Overfitting

A versão original do modelo apresentava overfitting severo (razão val/train de ~16x). As seguintes mudanças foram aplicadas:

| Melhoria | Detalhe |
| -------- | ------- |
| Data Augmentation | RandomFlip + GaussianNoise como camadas Keras (só ativas no treino) |
| Modelo menor | Filtros CNN: 16→32→64 reduzidos para 8→16→32; Dense: 128→64 reduzidos para 64→32 |
| Dropout aumentado | 0.35→0.50 e 0.20→0.30 |
| Regularização L2 | `l2(1e-4)` nas camadas Dense |
| K-Fold CV | Avaliação por 5-fold ao final do treino para estimativa robusta de generalização |

## Hiperparâmetros usados

| Parâmetro        | Valor                            |
| ---------------- | -------------------------------- |
| Otimizador       | Adam                             |
| Learning Rate    | Auto + ReduceLROnPlateau         |
| Batch Size       | 16                               |
| Loss             | MSE                              |
| Métrica          | MAE                              |
| Epochs máx       | 400                              |
| EarlyStopping    | patience = 60                    |
| Divisão de dados | 70% treino / 15% val / 15% teste |

Modelos salvos em:
results/models/cnn_pca_model.keras
results/models/history.npy

---

# 7. Reconstrução das Curvas e Comparação (`compare_curves.py`)

Para cada amostra:

1. Prediz os 30 componentes PCA
2. Aplica `pca.inverse_transform`
3. Desfaz normalização min–max da curva original
4. Gera gráfico SIM (simulado) vs DL (modelo)

Saídas em:
results/comparisons/comparison_ID_angulo.png

---

# 8. Análise de Resultados (`analyze_results.py`)

O script plota automaticamente:

### ✔ Histórico de Loss (treino vs validação)

Mostra convergência, estabilidade e ausência de overfitting grave.

### ✔ Histórico de MAE (treino vs validação)

Mostra o erro absoluto médio ao longo das épocas.

### ✔ MAE por comprimento de onda

Mostra quais regiões do espectro são mais difíceis de prever  
(os maiores erros ocorrem tipicamente nos picos acima de 600–800 nm).

---

# Resultado Geral

- O modelo generaliza corretamente para estruturas não vistas durante o treino.
- Razão val/train de ~1.33x, indicando ausência de overfitting severo.
- O erro residual (~MAE 0.15 no espaço PCA) é explicado pelo tamanho limitado do dataset — quanto mais estruturas simuladas forem adicionadas, menor o erro esperado.
- Os maiores erros ocorrem em picos de absorção agudos na região 600–800 nm, onde a variabilidade entre estruturas é maior.

---

# Parâmetros Principais (Resumo)

## Dados

- Estruturas usadas:  
  `1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24`
- Ângulos: 6 por estrutura
- Total: **126 amostras**

## Imagens

- 128×128 px
- 1 canal (grayscale)

## Curvas

- 101 pontos por curva

## PCA

- 30 componentes
- Variância explicada ~100%

## Arquitetura

- Data Augmentation (RandomFlip + GaussianNoise)
- CNN (8→16→32 filtros)
- MLP (64 → 32 → 30)
- Dropout (0.5, 0.3) + L2

## Treinamento

- Batch 16
- EarlyStopping (patience=60)
- ReduceLROnPlateau
- MSE + MAE
- K-Fold CV (k=5) para avaliação final

---

# Conclusão

O pipeline completo realiza:

1. **Geração automática** de imagens das geometrias 2D rotacionadas
2. **Processamento limpo** dos espectros simulados
3. **Normalização min–max** curva a curva
4. **Compressão PCA (30 componentes)**
5. **Treinamento CNN+MLP** com regularização robusta para prever os componentes PCA
6. **Reconstrução das curvas** e comparação com os espectros do COMSOL
7. **Análise detalhada de erros e estabilidade do modelo**

O resultado é um sistema capaz de prever curvas de absorção com boa precisão e custo computacional mínimo, sem overfitting severo.

---

# Execução do Pipeline

Use:

```bash
./run_pipeline.sh
```
