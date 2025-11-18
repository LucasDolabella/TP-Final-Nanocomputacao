#!/usr/bin/env bash
set -e  # para o script parar se qualquer etapa falhar

echo "=== Ativando ambiente virtual (.venv) ==="
# Se o script estiver na raiz do projeto
source .venv/bin/activate

echo
echo "=== 1/5 - Gerando geometrias e espectros (geometries.py) ==="
python geometries.py

echo
echo "=== 2/5 - Montando dataset (build_dataset.py) ==="
python build_dataset.py

echo
echo "=== 3/5 - Treinando modelo (train_model.py) ==="
python train_model.py

echo
echo "=== 4/5 - Gerando comparações SIM vs DL (compare_curves.py) ==="
python compare_curves.py

echo
echo "=== 5/5 - Gerando análises de treinamento (analyze_training.py) ==="
python analyze_results.py

echo
echo "✅ Pipeline completo!"
echo "   - Imagens das geometrias: results/geometries"
echo "   - Espectros:             results/spectrum"
echo "   - Dataset preparado:     results/prepared"
echo "   - Modelo treinado:       results/models"
echo "   - Comparações curvas:    results/comparisons"
echo "   - Análises dos resultados:    results/analysis"

# para rodar 
# chmod +x run_pipeline.sh
# ./run_pipeline.sh
