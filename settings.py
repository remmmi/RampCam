#!/usr/bin/env python3
"""Lecture de settings.conf (configuration centrale).

Usage shell : python3 app/settings.py <cle>  -> affiche le chemin de la section [paths].
"""
import configparser
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.conf")

_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
if not _config.read(SETTINGS_PATH, encoding="utf-8"):
    raise SystemExit(f"settings.conf introuvable : {SETTINGS_PATH}")


def path(key: str) -> str:
    """Chemin absolu de la cle `key` de la section [paths]."""
    return os.path.join(APP_DIR, os.path.expanduser(_config["paths"][key]))


DATA_DIR = path("data_dir")
CAPTURES_DIR = path("captures_dir")
LOGS_DIR = path("logs_dir")
ANNOTATED_DIR = path("annotated_dir")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage : settings.py <cle de [paths]>")
    print(path(sys.argv[1]))
