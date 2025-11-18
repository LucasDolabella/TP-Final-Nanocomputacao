#!/usr/bin/env bash
set -e  # para o script se qualquer etapa der erro

echo "=== Ativando ambiente virtual (.venv) ==="
# se o script estiver na raiz do projeto:
source .venv/bin/activate

echo
echo "=== 1/4 - Gerando geometrias e espectros (geometries.py) ==="
python geometries.py

echo
echo "=== 2/4 - Montando dataset (build_dataset.py) ==="
python build_dataset.py

echo
echo "=== 3/4 - Treinando modelo (train_model.py) ==="
python train_model.py

echo
echo "=== 4/4 - Gerando comparações SIM vs DL (compare_curves.py) ==="
python compare_curves.py

echo
echo "✅ Pipeline completo!"
echo "   - Imagens das geometrias: results/geometries"
echo "   - Espectros:             results/spectrum"
echo "   - Dataset preparado:     results/prepared"
echo "   - Modelo treinado:       results/models"
echo "   - Comparações curvas:    results/comparisons"

# para rodar 
# chmod +x run_pipeline.sh
# ./run_pipeline.sh
