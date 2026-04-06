# Predição de Curvas de Absorção a partir de Geometrias 2D de Nanoestruturas

## Objetivo

Nanoestruturas fotônicas têm propriedades ópticas (como absorção de luz) que dependem diretamente da sua geometria. Obter essas propriedades normalmente exige simulações físicas no COMSOL — software que pode levar horas por estrutura.

Este projeto desenvolve um modelo de **deep learning** que aprende a prever a **curva de absorção espectral** de uma nanoestrutura diretamente a partir da sua **imagem 2D**, eliminando a necessidade de simulação para estruturas novas.

A abordagem combina:
- **CNN** para extrair features da geometria
- **PCA** para comprimir e reconstruir curvas espectrais
- **MLP** para mapear features visuais → componentes PCA

---

## Estrutura do Projeto

```
TP-Final-Nanocomputacao/
├── dataset/
│   ├── input/          # Imput_XX.txt — coordenadas dos polígonos
│   └── output/         # Output_XX.txt — espectros simulados (COMSOL)
├── results/
│   ├── geometries/     # Imagens 128×128 das geometrias rotacionadas
│   ├── spectrum/       # Gráficos dos espectros simulados
│   ├── prepared/       # Arrays numpy prontos para treino
│   ├── models/         # Modelo treinado + histórico
│   ├── comparisons/    # Gráficos SIM vs DL
│   └── analysis/       # Gráficos de erro e histórico de treino
├── geometries.py       # Passo 1: gera imagens das geometrias
├── build_dataset.py    # Passo 2: prepara dataset (PCA, normalização)
├── train_model.py      # Passo 3: treina o modelo CNN+PCA
├── compare_curves.py   # Passo 4: gera comparações SIM vs DL
├── analyze_results.py  # Passo 5: análise de erros e histórico
└── run_pipeline.sh     # Executa o pipeline completo
```

---

## Como Executar

```bash
./run_pipeline.sh
```

Isso executa os 5 passos em ordem. Para rodar individualmente:

```bash
python geometries.py
python build_dataset.py
python train_model.py
python compare_curves.py
python analyze_results.py
```

---

## Pipeline Detalhado

### Passo 1 — Geração das Imagens (`geometries.py`)

**Entrada:** `dataset/input/Imput_XX.txt` — arquivo com as coordenadas (x, y) do polígono que define a geometria da nanoestrutura.

**O que o script faz:**

1. Lê as coordenadas do polígono
2. Centraliza a geometria na origem
3. Aplica uma rotação para cada um dos 6 ângulos simulados: **0°, 15°, 30°, 45°, 60°, 75°**
4. Normaliza a escala para que todas as estruturas caibam no mesmo campo visual (raio máximo × 1.05)
5. Renderiza uma imagem **128×128 pixels** em escala de cinza
6. Binariza: pixels dentro da geometria ficam **pretos (0)**, fora ficam **brancos (255)**
7. Salva a imagem em `results/geometries/geom_ID_angulo.png`

**Por que 6 ângulos?**  
Os dados do COMSOL incluem simulações para cada ângulo de incidência da luz. Usar todos os ângulos multiplica o dataset por 6 (de 21 para 126 amostras), o que é essencial dado o tamanho reduzido do conjunto.

**Saída:** 126 imagens (21 estruturas × 6 ângulos)

---

### Passo 2 — Construção do Dataset (`build_dataset.py`)

**Entrada:** Imagens geradas no Passo 1 + arquivos `dataset/output/Output_XX.txt`

**O que o script faz:**

#### 2.1 Leitura dos espectros

Para cada estrutura e ângulo:
- Lê o arquivo de saída do COMSOL
- Filtra as linhas correspondentes ao ângulo desejado
- Ordena pelos comprimentos de onda
- Extrai os 101 pontos da curva de absorção

#### 2.2 Normalização min-max por curva

Cada curva é normalizada individualmente para o intervalo [0, 1]:

```
y_norm = (y - y_min) / (y_max - y_min)
```

Isso evita que curvas com amplitudes muito diferentes dominem o treinamento.  
Os valores `y_min` e `y_max` de cada curva são salvos para a desnormalização posterior.

#### 2.3 Compressão PCA

As 101 dimensões de cada curva normalizada são comprimidas para **30 componentes PCA**.

- **Variância explicada: ~99.9997%** — a compressão é praticamente sem perda
- PCA reduz o problema de regressão de 101 saídas para 30, tornando o treino mais estável
- O modelo PCA é salvo em `pca_model.pkl` para reconstrução posterior

**Por que PCA?**  
Curvas espectrais têm alta correlação entre comprimentos de onda vizinhos — a maior parte da informação está nos primeiros componentes. Isso permite ao modelo focar no que realmente importa em vez de prever 101 valores independentes.

**Saídas geradas em `results/prepared/`:**

| Arquivo | Shape | Descrição |
|---------|-------|-----------|
| `X_images.npy` | (126, 128, 128, 1) | Imagens normalizadas [0,1] |
| `Y_raw.npy` | (126, 101) | Curvas originais (escala física) |
| `Y_pca.npy` | (126, 30) | Curvas comprimidas (componentes PCA) |
| `metadata.npy` | (126, 2) | [ID da estrutura, ângulo em graus] |
| `wavelengths.npy` | (101,) | Eixo de comprimentos de onda (nm) |
| `curve_min.npy` | (126,) | Mínimo de cada curva (desnormalização) |
| `curve_max.npy` | (126,) | Máximo de cada curva (desnormalização) |
| `pca_model.pkl` | — | Modelo PCA treinado |

---

### Passo 3 — Treinamento do Modelo (`train_model.py`)

**Entrada:** `X_images.npy` e `Y_pca.npy`

**Divisão dos dados:**
- 70% treino (~88 amostras)
- 15% validação (~19 amostras)
- 15% teste (~19 amostras)

#### Arquitetura do Modelo (CNN + MLP)

```
Entrada: imagem 128×128×1
    │
    ├─ RandomFlip (horizontal e vertical)       ← augmentation, só no treino
    ├─ GaussianNoise (σ = 0.05)                ← augmentation, só no treino
    │
    ├─ Conv2D(8,  3×3, relu) + L2(1e-4)
    ├─ MaxPooling2D(2×2)
    │
    ├─ Conv2D(16, 3×3, relu) + L2(1e-4)
    ├─ MaxPooling2D(2×2)
    │
    ├─ Conv2D(32, 3×3, relu) + L2(1e-4)
    ├─ MaxPooling2D(2×2)
    │
    ├─ Flatten
    │
    ├─ Dense(64, relu) + L2(1e-4) + Dropout(0.5)
    ├─ Dense(32, relu) + L2(1e-4) + Dropout(0.3)
    │
    └─ Dense(30, linear)    ← 30 componentes PCA preditos
```

#### Estratégia Anti-Overfitting

Com apenas ~88 amostras de treino, o risco de overfitting é alto. As seguintes técnicas foram aplicadas em conjunto:

| Técnica | Configuração | Motivação |
|---------|-------------|-----------|
| **Data Augmentation — RandomFlip** | Espelha geometrias horizontalmente e verticalmente | Dobra/quadruplica a diversidade das amostras |
| **Data Augmentation — GaussianNoise** | σ = 0.05 (aumentado de 0.02) | Geometrias são binárias (0/1); ruído maior força robustez sem distorcer o sinal |
| **L2 nas camadas Conv2D** | `kernel_regularizer=l2(1e-4)` | Penaliza filtros com pesos muito grandes, evitando especialização excessiva nas imagens de treino |
| **L2 nas camadas Dense** | `kernel_regularizer=l2(1e-4)` | Mesma lógica aplicada ao regressor MLP |
| **Dropout(0.5)** | Na primeira camada Dense | Desativa 50% dos neurônios aleatoriamente por batch, forçando redundância |
| **Dropout(0.3)** | Na segunda camada Dense | Regularização mais suave próximo à saída |
| **Modelo compacto** | 8→16→32 filtros, Dense 64→32 | Reduz a capacidade de memorização sem perder poder de representação |

> **Nota sobre o Global Average Pooling (GAP):** foi testado como substituto do Flatten.
> Embora elimine o overfitting quase completamente (Val/Train 1.01x), o modelo ficou
> sub-representado — a validação piorou em valores absolutos (MSE 0.142 vs 0.135 com Flatten).
> O Flatten com L2 nas Conv apresenta melhor equilíbrio entre capacidade e generalização.

#### Hiperparâmetros de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Otimizador | Adam |
| Loss | MSE |
| Métrica | MAE |
| Batch size | 16 |
| Épocas máximas | 400 |
| EarlyStopping | patience = 60 (restaura melhores pesos) |
| ReduceLROnPlateau | fator 0.5, patience = 15, mín = 1e-5 |

#### Avaliação Final por K-Fold

Além do treino principal, o script realiza **K-Fold Cross-Validation (k=5)** sobre todo o dataset para uma estimativa robusta de generalização, independente da divisão aleatória treino/teste.

**Resultados obtidos:**

| Conjunto | MSE | MAE |
|----------|-----|-----|
| Treino | 0.1053 | 0.1419 |
| Validação | 0.1350 | 0.1555 |
| Teste | 0.1417 | 0.1535 |
| **CV (k=5)** | **0.1488 ± 0.0125** | **0.1596 ± 0.0022** |

**Razão Val/Train: 1.28x** — overfitting moderado e controlado, esperado para 88 amostras.

**Saídas:**
- `results/models/cnn_pca_model.keras` — modelo treinado
- `results/models/history.npy` — histórico de loss e MAE por época

---

### Passo 4 — Comparação das Curvas (`compare_curves.py`)

**O que o script faz:**

Para cada amostra selecionada:

1. Carrega a imagem da geometria
2. Passa pelo modelo CNN → obtém 30 componentes PCA preditos
3. Aplica `pca.inverse_transform` → reconstrói 101 pontos normalizados
4. Desfaz a normalização min-max → curva na escala física original
5. Plota **SIM** (curva do COMSOL) vs **DL** (curva predita)

O script replica a mesma divisão aleatória do `train_model.py` (`random_state=42`) e gera comparações **exclusivamente para as 19 amostras do conjunto de teste** — estruturas que o modelo nunca viu durante o treino. Isso garante que os gráficos reflitam capacidade real de generalização, não memorização.

**Por que só o conjunto de teste?**  
Das 126 amostras, ~88 foram usadas no treino. Gerar comparações para essas produziria curvas artificialmente boas, pois o modelo já as "memorizou" parcialmente. Usar só o teste é a avaliação cientificamente honesta.

**Saídas:** `results/comparisons/comparison_ID_angulo.png` (19 arquivos)

#### Interpretação dos Resultados

- **Picos largos (baixa frequência espectral):** bem capturados pelo modelo — a posição e forma geral são reproduzidas com boa fidelidade.
- **Picos estreitos/agudos (ressonâncias em 650–750 nm):** subestimados em amplitude e largura. Isso é esperado: picos agudos concentram energia em poucos comprimentos de onda e dependem de componentes PCA de menor variância, que são mais difíceis de prever com poucos dados de treino.

---

### Passo 5 — Análise de Resultados (`analyze_results.py`)

Gera três gráficos de diagnóstico em `results/analysis/`:

#### `history_loss.png` — Histórico da Loss (MSE)

Mostra a evolução do MSE ao longo das épocas para treino e validação.  
**O que observar:** as duas curvas devem descer juntas e permanecer próximas. Um gap crescente indica overfitting.

#### `history_mae.png` — Histórico do MAE

Mostra o Erro Absoluto Médio (em unidades normalizadas) ao longo do treino.  
**O que observar:** convergência estável. Na versão atual, o MAE de validação ficou ligeiramente abaixo do de treino em vários momentos — sinal de boa generalização.

#### `mae_by_wavelength.png` — MAE por Comprimento de Onda

Mostra o erro médio do modelo em cada ponto do espectro.  
**O que observar:** o erro cresce consistentemente de 300 nm (baixo, ~0.08×10⁻¹⁴) para 800 nm (alto, ~1.1×10⁻¹⁴). Isso reflete diretamente a dificuldade de prever as ressonâncias plasmônicas agudas que ocorrem na região 600–800 nm.

---

## Dados Utilizados

- **21 estruturas** simuladas no COMSOL
- **6 ângulos de incidência** por estrutura: 0°, 15°, 30°, 45°, 60°, 75°
- **126 amostras totais**
- Estruturas: `1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24`

---

## Limitações e Discussão

**Por que o modelo não captura picos agudos perfeitamente?**

A limitação principal é o tamanho do dataset. Com ~88 amostras de treino:
- O modelo aprende a prever o padrão médio das curvas (picos largos, tendências gerais)
- Picos estreitos exigem que o modelo generalize a partir de poucos exemplos de cada tipo de ressonância — tarefa estatisticamente difícil

Isso **não é uma limitação da arquitetura** — é uma limitação de dados, comum em problemas de nanofotônica onde cada simulação tem custo computacional significativo.

**O que melhoraria os resultados:**
1. Mais estruturas simuladas (50–100 estruturas → dataset de 300–600 amostras)
2. Augmentation físico (pequenas perturbações geométricas das estruturas existentes)
3. Transfer learning a partir de geometrias sintéticas

---

## Dependências

```
tensorflow
numpy
scikit-learn
matplotlib
joblib
Pillow
```

Instalação:

```bash
pip install -r requirements.txt
```
