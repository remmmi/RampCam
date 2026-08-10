#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$HOME/Bureau/cam-venv-headless"

if [[ ! -d "$VENV_DIR" ]]; then
  cat <<'EOF'
[ERROR] Environnement headless introuvable: ~/Bureau/cam-venv-headless
Créez-le avec:

python3 -m venv ~/Bureau/cam-venv-headless
source ~/Bureau/cam-venv-headless/bin/activate
pip install --upgrade pip
pip install --no-cache-dir opencv-python-headless==4.11.0.86 numpy pillow requests
python -c "import cv2; print(cv2.__version__); print('dnn:', hasattr(cv2,'dnn'))"

deactivate

Puis relancez ce script.
EOF
  exit 1
fi

source "$VENV_DIR/bin/activate"

if [[ $# -gt 0 ]]; then
  # Exécuter la commande fournie dans le venv
  "$@"
else
  # Commande par défaut: classer tout le dossier captures en headless et sauvegarder les vidéos annotées
  python3 app/classify.py app/captures/ -r --conf 0.35 --nms 0.45 --width 640 --no-display --save --out-dir app/captures2
fi

deactivate
