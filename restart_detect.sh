#!/bin/bash
# Script pour redémarrer la détection avec le masque

echo "Arrêt du script de détection en cours..."
pkill -f detect.py
sleep 2

echo "Démarrage du script de détection avec masque..."
cd "$(dirname "$0")/.."
./bin/python3 app/detect.py --sensitivity 100 --min-area 500 &

echo "Script de détection redémarré avec PID: $!"
echo "Logs disponibles dans: $(./bin/python3 app/settings.py logs_dir)/$(date +%Y.%m.%d).detect.log"
