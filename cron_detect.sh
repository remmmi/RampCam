#!/bin/bash
# Watchdog cron : verifie que detect.py tourne, le relance sinon.
# Donne l'acces a l'affichage X (fenetre video visible).
# Appele chaque minute par cron + au demarrage (@reboot).

# --- Acces au serveur X de la session ---
export DISPLAY=:0
export XAUTHORITY=/home/m/.Xauthority

# Racine du projet deduite de l'emplacement du script (survit aux deplacements)
VENV="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$VENV/bin/python3"
SCRIPT="$VENV/app/detect.py"
LOG_DIR="$("$PYTHON" "$VENV/app/settings.py" logs_dir)" || exit 1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watchdog.log"
LOCK=/tmp/cron_detect.lock

# Evite deux executions simultanees du watchdog
exec 9>"$LOCK"
flock -n 9 || exit 0

# Si le serveur X n'est pas (encore) pret, on ne tente rien.
[ -S /tmp/.X11-unix/X0 ] || exit 0

# Deja en cours ? -> rien a faire.
if pgrep -f "python3 .*app/detect.py" >/dev/null; then
    exit 0
fi

# Sinon, (re)lancement detache, avec acces a l'affichage.
cd "$VENV" || exit 1
echo "$(date '+%d/%m/%Y %H:%M:%S') [cron] relance detect.py" >> "$LOG"
setsid "$PYTHON" "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
