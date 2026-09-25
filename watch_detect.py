#!/usr/bin/env python3
import subprocess
import time
import os
import signal
from settings import LOGS_DIR

SCRIPT_NAME = "app/detect.py"
# Racine du projet deduite de l'emplacement du script (survit aux deplacements)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_PATH = os.path.join(ROOT, "bin/python3")
SCRIPT_PATH = os.path.join(ROOT, "app/detect.py")
LOG_PATH = os.path.join(LOGS_DIR, "watchdog.log")
os.makedirs(LOGS_DIR, exist_ok=True)

def is_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", SCRIPT_NAME])
        return True
    except subprocess.CalledProcessError:
        return False

def launch():
    env = os.environ.copy()
    env["DISPLAY"] = ":0"  # à adapter selon ta session X11
    subprocess.Popen(
        [PYTHON_PATH, SCRIPT_PATH],
        stdout=open(LOG_PATH, "a"),
        stderr=subprocess.STDOUT,
        env=env,
    )

while True:
    if not is_running():
        launch()
    time.sleep(60)
