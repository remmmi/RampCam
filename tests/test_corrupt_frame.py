#!/usr/bin/env python3
"""Tests autonomes pour is_corrupted_frame. Lancer avec le venv :
    ./bin/python3 app/tests/test_corrupt_frame.py
"""
import os
import sys
import numpy as np

# Permet d'importer detect.py (situé dans app/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import detect


def _solid(bgr):
    """Crée une frame 480x640 d'une couleur BGR unie."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_neon_green_is_corrupt():
    assert detect.is_corrupted_frame(_solid((0, 255, 0))) is True


def test_magenta_is_corrupt():
    assert detect.is_corrupted_frame(_solid((255, 0, 255))) is True


def test_muted_gray_is_clean():
    assert detect.is_corrupted_frame(_solid((120, 120, 120))) is False


def test_small_neon_patch_below_threshold_is_clean():
    # ~2% de l'image en vert néon -> sous le seuil de 5%
    frame = _solid((120, 120, 120))
    # 480*640 = 307200 px ; 2% ≈ 6144 px -> bande de 10 lignes x 640
    frame[0:10, :] = (0, 255, 0)
    assert detect.is_corrupted_frame(frame) is False


def test_large_neon_patch_above_threshold_is_corrupt():
    # ~10% de l'image en magenta -> au-dessus du seuil de 5%
    frame = _solid((120, 120, 120))
    frame[0:48, :] = (255, 0, 255)  # 48/480 = 10%
    assert detect.is_corrupted_frame(frame) is True


def test_error_returns_false_fail_open():
    # Une entrée invalide ne doit jamais aveugler la caméra
    assert detect.is_corrupted_frame(None) is False


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
