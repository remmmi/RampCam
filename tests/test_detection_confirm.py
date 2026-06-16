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


def test_confirm_failsafe_when_yolo_unavailable():
    # Si YOLO indisponible, on laisse passer (fail-safe) sans toucher au réseau.
    saved = detect.yolo_net
    detect.yolo_net = None
    try:
        confirmed, label, conf = detect.confirm_interesting_object("url", ("a", "b"), None)
        assert confirmed is True
        assert label is None
        assert conf == 0.0
    finally:
        detect.yolo_net = saved


def test_best_interesting_uses_objectness():
    # voiture (classe 2) : proba classe 0.50 mais objectness faible 0.36 -> conf = 0.18
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 4] = 0.36       # objectness
    det[0, 5 + 2] = 0.50   # proba 'voiture'
    class_id, conf = detect._best_interesting([det], 640, 480)
    assert class_id == 2
    assert abs(conf - 0.18) < 1e-3
    # donc rejeté par la règle de décision (seuil 0.4)
    assert detect.is_interesting_detection(class_id, conf) is False


def test_best_interesting_strong_detection():
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 4] = 0.9
    det[0, 5 + 2] = 0.8
    class_id, conf = detect._best_interesting([det], 640, 480)
    assert class_id == 2
    assert abs(conf - 0.72) < 1e-3


def test_best_interesting_rejects_detection_out_of_mask():
    # détection forte mais centre hors zone surveillée -> ignorée
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 0] = 0.95       # centre x (haut-droite)
    det[0, 1] = 0.2        # centre y
    det[0, 4] = 0.9
    det[0, 5 + 2] = 0.9
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[240:480, 0:320] = 255   # zone surveillée = bas-gauche
    class_id, conf = detect._best_interesting([det], 640, 480, mask)
    assert class_id is None
    assert conf == 0.0


def test_best_interesting_keeps_detection_in_mask():
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 0] = 0.25       # centre x (bas-gauche)
    det[0, 1] = 0.75       # centre y
    det[0, 4] = 0.9
    det[0, 5 + 2] = 0.9
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[240:480, 0:320] = 255
    class_id, conf = detect._best_interesting([det], 640, 480, mask)
    assert class_id == 2
    assert abs(conf - 0.81) < 1e-3


def test_detect_interesting_blank_frame_no_object():
    # Une frame noire ne contient aucun objet -> (None, 0.0).
    if detect.yolo_net is None:
        print("  (smoke ignoré : yolo non chargé)")
        return
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    class_id, conf = detect._detect_interesting(frame)
    assert class_id is None
    assert conf == 0.0


def test_confirm_returns_on_first_interesting_frame():
    # La 1re frame contient une classe -> confirmé sans aucun refetch.
    saved = (detect.yolo_net, detect.is_corrupted_frame,
             detect._detect_interesting, detect.fetch_image)
    detect.yolo_net = object()
    detect.is_corrupted_frame = lambda f: False
    detect._detect_interesting = lambda f, m=None: (0, 0.9)  # 0 = personne
    calls = {"fetch": 0}

    def fake_fetch(u, a):
        calls["fetch"] += 1
        return "good"

    detect.fetch_image = fake_fetch
    try:
        result = detect.confirm_interesting_object("u", ("a", "b"), "good")
        assert result == (True, "personne", 0.9)
        assert calls["fetch"] == 0
    finally:
        (detect.yolo_net, detect.is_corrupted_frame,
         detect._detect_interesting, detect.fetch_image) = saved


def test_confirm_skips_corrupted_without_consuming_budget():
    # Les frames corrompues ne consomment pas le budget d'analyse (YOLO_CONFIRM_FRAMES).
    saved = (detect.yolo_net, detect.is_corrupted_frame, detect._detect_interesting,
             detect.fetch_image, detect.YOLO_CONFIRM_INTERVAL)
    detect.yolo_net = object()
    detect.is_corrupted_frame = lambda f: f == "bad"
    analyse = {"n": 0}

    def fake_detect(f, m=None):
        analyse["n"] += 1
        return (None, 0.0)  # jamais de classe intéressante

    detect._detect_interesting = fake_detect
    seq = ["bad", "good", "bad", "good", "bad", "good"]
    idx = {"i": 0}

    def fake_fetch(u, a):
        v = seq[idx["i"]] if idx["i"] < len(seq) else "good"
        idx["i"] += 1
        return v

    detect.fetch_image = fake_fetch
    detect.YOLO_CONFIRM_INTERVAL = 0  # pas d'attente réelle pendant le test
    try:
        result = detect.confirm_interesting_object("u", ("a", "b"), "good")
        assert result == (False, None, 0.0)
        assert analyse["n"] == detect.YOLO_CONFIRM_FRAMES
    finally:
        (detect.yolo_net, detect.is_corrupted_frame, detect._detect_interesting,
         detect.fetch_image, detect.YOLO_CONFIRM_INTERVAL) = saved


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
