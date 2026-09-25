#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$HOME/Bureau/cam-venv-gui"

if [[ ! -d "$VENV_DIR" ]]; then
  cat <<'EOF'
[ERROR] Environnement GUI introuvable: ~/Bureau/cam-venv-gui
Créez-le avec (Ubuntu/Debian):

sudo apt-get update
sudo apt-get install -y libgtk-3-0 libgtk-3-dev python3-tk ffmpeg

python3 -m venv ~/Bureau/cam-venv-gui
source ~/Bureau/cam-venv-gui/bin/activate
pip install --upgrade pip
pip install --no-cache-dir opencv-python==4.11.0.86 numpy pillow requests
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
  # Commande par défaut: lecture fenêtrée des vidéos du dossier captures
  python3 app/classify.py "$(python3 app/settings.py captures_dir)" -r --conf 0.35 --nms 0.45 --width 640
fi

deactivate
