#!/usr/bin/env python3
"""Tests autonomes pour la détection à 2 étages. Lancer avec le venv :
    ./bin/python3 app/tests/test_detection_confirm.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import detect


def test_is_interesting_detection_person_above_threshold():
    # classe 0 = personne, présente dans INTERESTING_CLASSES
    assert detect.is_interesting_detection(0, 0.9) is True


def test_is_interesting_detection_below_threshold():
    assert detect.is_interesting_detection(0, 0.1) is False


def test_is_interesting_detection_uninteresting_class():
    # classe 10 (feu tricolore) absente de INTERESTING_CLASSES
    assert detect.is_interesting_detection(10, 0.99) is False


def test_is_interesting_detection_threshold_is_inclusive():
    # à exactement le seuil par défaut -> confirmé
    assert detect.is_interesting_detection(0, detect.YOLO_CONFIRM_CONF) is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} OK")
    sys.exit(1 if failed else 0)
