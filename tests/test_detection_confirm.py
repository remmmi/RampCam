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
    class_id, conf, _ = detect._best_interesting([det], 640, 480)
    assert class_id == 2
    assert abs(conf - 0.18) < 1e-3
    # donc rejeté par la règle de décision (seuil 0.4)
    assert detect.is_interesting_detection(class_id, conf) is False


def test_best_interesting_strong_detection():
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 4] = 0.9
    det[0, 5 + 2] = 0.8
    class_id, conf, _ = detect._best_interesting([det], 640, 480)
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
    class_id, conf, _ = detect._best_interesting([det], 640, 480, mask)
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
    class_id, conf, _ = detect._best_interesting([det], 640, 480, mask)
    assert class_id == 2
    assert abs(conf - 0.81) < 1e-3


def test_detect_interesting_blank_frame_no_object():
    # Une frame noire ne contient aucun objet -> (None, 0.0).
    if detect.yolo_net is None:
        print("  (smoke ignoré : yolo non chargé)")
        return
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    class_id, conf, _ = detect._detect_interesting(frame)
    assert class_id is None
    assert conf == 0.0


def test_confirm_returns_on_first_interesting_frame():
    # La 1re frame contient une classe -> confirmé sans aucun refetch.
    saved = (detect.yolo_net, detect.is_corrupted_frame,
             detect._detect_interesting, detect.fetch_image)
    detect.yolo_net = object()
    detect.is_corrupted_frame = lambda f: False
    detect._detect_interesting = lambda f, m=None, mb=None: (0, 0.9, (10, 10, 50, 50))  # 0 = personne
    calls = {"fetch": 0}

    def fake_fetch(u, a):
        calls["fetch"] += 1
        return "good"

    detect.fetch_image = fake_fetch
    detect.static_objects.clear()
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

    def fake_detect(f, m=None, mb=None):
        analyse["n"] += 1
        return (None, 0.0, None)  # jamais de classe intéressante

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


def _car_detection():
    # voiture au centre : boîte pixels (256, 192, 128, 96), conf 0.81
    det = np.zeros((1, 5 + 80), dtype=np.float32)
    det[0, 0], det[0, 1], det[0, 2], det[0, 3] = 0.5, 0.5, 0.2, 0.2
    det[0, 4] = 0.9
    det[0, 5 + 2] = 0.9
    return [det]


def test_best_interesting_rejects_object_outside_motion():
    # objet statique : mouvement ailleurs dans l'image -> ignoré
    class_id, conf, _ = detect._best_interesting(_car_detection(), 640, 480,
                                                 motion_boxes=[(500, 50, 60, 60)])
    assert class_id is None
    assert conf == 0.0


def test_best_interesting_accepts_object_overlapping_motion():
    class_id, conf, box = detect._best_interesting(_car_detection(), 640, 480,
                                                   motion_boxes=[(300, 220, 50, 50)])
    assert class_id == 2
    assert abs(conf - 0.81) < 1e-3
    assert box == (256, 192, 128, 96)


def test_best_interesting_without_motion_boxes_backcompat():
    class_id, conf, _ = detect._best_interesting(_car_detection(), 640, 480, motion_boxes=None)
    assert class_id == 2


def test_motion_boxes_filters_area_adds_margin_and_clamps():
    m = detect.MOTION_BOX_MARGIN_PX
    carre_20 = np.array([[[100, 100]], [[120, 100]], [[120, 120]], [[100, 120]]])  # aire 400
    carre_5 = np.array([[[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]])           # aire 25
    # boundingRect est inclusif : points 100..120 -> largeur 21
    boxes = detect._motion_boxes([carre_20, carre_5], min_area=400, w=640, h=480)
    assert boxes == [(100 - m, 100 - m, 21 + 2 * m, 21 + 2 * m)]
    coin = np.array([[[0, 0]], [[30, 0]], [[30, 30]], [[0, 30]]])                  # aire 900
    assert detect._motion_boxes([coin], min_area=400, w=640, h=480) == [(0, 0, 31 + m, 31 + m)]


def test_iou_values():
    assert abs(detect._iou((0, 0, 100, 100), (0, 0, 100, 100)) - 1.0) < 1e-6
    assert detect._iou((0, 0, 50, 50), (100, 100, 50, 50)) == 0.0
    assert abs(detect._iou((0, 0, 100, 100), (50, 0, 100, 100)) - 1 / 3) < 1e-3


def test_static_check_suppresses_from_second_confirmation():
    reg = []
    car_box = (256, 192, 128, 96)
    assert detect._static_check(reg, 2, car_box, now=1000.0) is False  # 1re alerte passe
    assert detect._static_check(reg, 2, car_box, now=1030.0) is True   # sourdine
    assert detect._static_check(reg, 2, car_box, now=1060.0) is True


def test_static_check_moved_object_alerts_again():
    reg = []
    car_box = (256, 192, 128, 96)
    detect._static_check(reg, 2, car_box, now=1000.0)
    detect._static_check(reg, 2, car_box, now=1030.0)
    assert detect._static_check(reg, 2, (450, 192, 128, 96), now=1100.0) is False


def test_static_check_other_class_same_spot_alerts():
    reg = []
    car_box = (256, 192, 128, 96)
    detect._static_check(reg, 2, car_box, now=1000.0)
    detect._static_check(reg, 2, car_box, now=1030.0)
    # une personne devant la voiture garée doit alerter (classe différente)
    assert detect._static_check(reg, 0, car_box, now=1040.0) is False


def test_static_check_forgets_after_expiry():
    reg = []
    car_box = (256, 192, 128, 96)
    detect._static_check(reg, 2, car_box, now=0.0)
    assert detect._static_check(reg, 2, car_box, now=30.0) is True
    late = 30.0 + detect.STATIC_FORGET_SECS + 1
    assert detect._static_check(reg, 2, car_box, now=late) is False


def test_warmup_active_window():
    assert detect._warmup_active(1000.0, 1000.0 + detect.STARTUP_WARMUP_SECS - 1) is True
    assert detect._warmup_active(1000.0, 1000.0 + detect.STARTUP_WARMUP_SECS + 1) is False


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
