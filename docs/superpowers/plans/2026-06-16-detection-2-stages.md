# Détection à 2 étages (MOG2 + confirmation YOLO) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réduire les faux déclenchements en ne lançant klaxon / enregistrement / ntfy que lorsqu'un mouvement (détecté par soustraction de fond MOG2) est confirmé par yolov4-tiny comme contenant une classe intéressante (personne, véhicule, animal).

**Architecture:** Dans `detection()`, le pipeline devient : frame -> filtre corrompu (déjà en place) -> MOG2 (avant-plan) + masque de surveillance -> si mouvement et hors cooldown -> confirmation YOLO en rafale sur quelques frames saines -> si classe confirmée -> klaxon + enregistrement + ntfy. MOG2 remplace la diff à 2 images. YOLO est déjà chargé au démarrage et déjà utilisé pour la sélection de frame ntfy ; on ajoute juste son rôle de garde du déclenchement.

**Tech Stack:** Python 3, OpenCV (`cv2`, MOG2 + `cv2.dnn`), NumPy. Venv racine `./bin/python3`. Tests = scripts autonomes lancés avec l'interpréteur du venv (pas de pytest).

**Décisions validées (brainstorming) :**
- Les 3 actions (klaxon, ntfy, conservation vidéo) ne se déclenchent QUE si YOLO confirme.
- MOG2 remplace la diff à 2 images.
- Confirmation YOLO en rafale (jusqu'à quelques frames), arrêt à la première classe confirmée.
- Chaque frame de la rafale passe par `is_corrupted_frame` ; une frame corrompue/None ne compte pas et on en récupère une autre, dans la limite d'une borne de tentatives.
- Fail-safe : si YOLO est indisponible (`yolo_net is None`) ou si l'inférence plante, on laisse passer (déclenche sur le seul mouvement) pour ne jamais rendre la caméra muette.

**Préliminaire :** Travailler sur une branche dédiée.
```bash
cd /home/m/Bureau/camera-venv/app
git checkout -b feature/detection-2-stages
```

---

### Task 1 : Constantes + règle de décision `is_interesting_detection`

**Files:**
- Create: `app/tests/test_detection_confirm.py`
- Modify: `app/detect.py` (constantes après le bloc MOG2/corrupt vers la ligne 67 ; fonction dans la section YOLO après `_compute_yolo_score`, ~ligne 299)

- [ ] **Step 1 : Écrire le test qui échoue**

Créer `app/tests/test_detection_confirm.py` :

```python
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
```

- [ ] **Step 2 : Lancer le test pour vérifier qu'il échoue**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_detection_confirm.py`
Expected: ERROR/FAIL sur tous (`AttributeError: module 'detect' has no attribute 'is_interesting_detection'` ou `... 'YOLO_CONFIRM_CONF'`).

- [ ] **Step 3 : Ajouter les constantes dans `detect.py`**

Juste après la ligne `CORRUPT_SAT_FRAC_MAX = 0.05` (~ligne 66, avant `TRIGGER_COOLDOWN_SECS`), ajouter :

```python

# Détection à 2 étages : fond adaptatif (MOG2) + confirmation YOLO
MOG2_HISTORY = 500            # nb de frames pour le modèle de fond
MOG2_VAR_THRESHOLD = 16       # seuil de variance MOG2 (défaut OpenCV)
YOLO_CONFIRM_CONF = 0.4       # confiance mini pour confirmer une classe
YOLO_CONFIRM_FRAMES = 3       # nb de frames saines à analyser au plus
YOLO_CONFIRM_INTERVAL = 0.2   # délai (s) entre frames de la rafale
YOLO_CONFIRM_MAX_FETCH = 6    # borne de tentatives (anti-boucle si glitch en rafale)
YOLO_CHECK_COOLDOWN_SECS = 3  # délai mini entre deux confirmations YOLO (mouvement non confirmé)
```

- [ ] **Step 4 : Implémenter `is_interesting_detection` dans `detect.py`**

Dans la section YOLO, juste après la fonction `_compute_yolo_score` (se termine ~ligne 299), ajouter :

```python
def is_interesting_detection(class_id, confidence, threshold=YOLO_CONFIRM_CONF):
    """Règle de décision : vrai si la classe est surveillée ET la confiance suffisante."""
    return class_id in INTERESTING_CLASSES and confidence >= threshold
```

- [ ] **Step 5 : Lancer le test pour vérifier qu'il passe**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_detection_confirm.py`
Expected: `4/4 OK`, exit 0.

- [ ] **Step 6 : Commit**

```bash
cd /home/m/Bureau/camera-venv/app
git add detect.py tests/test_detection_confirm.py
git commit -m "feat: constantes 2 etages + regle is_interesting_detection"
```

---

### Task 2 : Inférence `_detect_interesting` + confirmation en rafale `confirm_interesting_object`

**Files:**
- Modify: `app/detect.py` (juste après `is_interesting_detection`, section YOLO)
- Modify: `app/tests/test_detection_confirm.py` (ajout de tests)

- [ ] **Step 1 : Ajouter les tests qui échouent**

Dans `app/tests/test_detection_confirm.py`, ajouter ces fonctions AVANT le bloc `if __name__ == "__main__":` :

```python
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


def test_detect_interesting_blank_frame_no_object():
    # Une frame noire ne contient aucun objet -> (None, 0.0).
    if detect.yolo_net is None:
        print("  (smoke ignoré : yolo non chargé)")
        return
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    class_id, conf = detect._detect_interesting(frame)
    assert class_id is None
    assert conf == 0.0
```

- [ ] **Step 2 : Lancer pour vérifier l'échec des nouveaux tests**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_detection_confirm.py`
Expected: les 4 tests de Task 1 PASS ; les 2 nouveaux en ERROR (`AttributeError: ... '_detect_interesting'` / `... 'confirm_interesting_object'`).

- [ ] **Step 3 : Implémenter les deux fonctions dans `detect.py`**

Juste après `is_interesting_detection` (ajoutée en Task 1), ajouter :

```python
def _detect_interesting(frame):
    """Renvoie (class_id, confiance) de la meilleure détection d'une classe surveillée,
    (None, 0.0) sinon. Ne capture PAS les exceptions : elles remontent pour le fail-safe."""
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(yolo_output_layers)
    best_conf = 0.0
    best_id = None
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if class_id in INTERESTING_CLASSES and confidence > best_conf:
                best_conf = confidence
                best_id = class_id
    return best_id, best_conf


def confirm_interesting_object(url, auth, first_frame):
    """Confirme la présence d'une classe surveillée via YOLO sur une rafale de frames.
    Renvoie (confirmé: bool, label: str|None, confiance: float).
    - Analyse d'abord first_frame (déjà filtrée), puis récupère des frames live.
    - Ignore les frames corrompues / None (ne comptent pas, on en refetch une autre).
    - Fail-safe : YOLO indisponible ou inférence en erreur -> (True, None, 0.0)."""
    if yolo_net is None:
        return True, None, 0.0
    frame = first_frame
    analysed = 0
    try:
        for _ in range(YOLO_CONFIRM_MAX_FETCH):
            if frame is not None and not is_corrupted_frame(frame):
                class_id, conf = _detect_interesting(frame)
                if class_id is not None and is_interesting_detection(class_id, conf):
                    return True, INTERESTING_CLASSES[class_id], conf
                analysed += 1
                if analysed >= YOLO_CONFIRM_FRAMES:
                    break
            time.sleep(YOLO_CONFIRM_INTERVAL)
            frame = fetch_image(url, auth)
        return False, None, 0.0
    except Exception as e:
        log(f"Confirmation YOLO: erreur, laisse passer - {e}")
        return True, None, 0.0
```

- [ ] **Step 4 : Lancer pour vérifier que tout passe**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_detection_confirm.py`
Expected: `6/6 OK`, exit 0.

- [ ] **Step 5 : Commit**

```bash
cd /home/m/Bureau/camera-venv/app
git add detect.py tests/test_detection_confirm.py
git commit -m "feat: confirmation YOLO en rafale (anti faux declenchements)"
```

---

### Task 3 : Rebrancher `detection()` sur MOG2 + garde de confirmation

**Files:**
- Modify: `app/detect.py` (fonction `detection`, ~lignes 548-617)

- [ ] **Step 1 : Remplacer l'init de `detection()`**

Le début actuel de `detection()` est :

```python
def detection(args):
    global last_trigger_time
    url = CAMERA_URL
    auth = CAMERA_AUTH
    delay = 1 / args.fps
    prev = None

    mask = load_surveillance_mask()
    mask_rs = None
```

Le remplacer par (suppression de `prev`, ajout du soustracteur de fond et du chrono de re-vérification) :

```python
def detection(args):
    global last_trigger_time
    url = CAMERA_URL
    auth = CAMERA_AUTH
    delay = 1 / args.fps
    last_motion_check = 0

    mask = load_surveillance_mask()
    mask_rs = None
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=True)
```

- [ ] **Step 2 : Remplacer le bloc de détection/déclenchement**

Le bloc actuel (de `gray = cv2.GaussianBlur(...)` jusqu'à `prev = gray` inclus, ~lignes 584-617) est :

```python
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if prev is None:
            prev = gray
            time.sleep(delay)
            continue

        delta = cv2.absdiff(prev, gray)
        thresh = cv2.dilate(cv2.threshold(delta, args.sensitivity, 255, cv2.THRESH_BINARY)[1],
                            None, iterations=2)

        if mask_rs is not None:
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask_rs)
            # Debug overlay : sauvegarde image réelle + masque coloré
            #debug_img = debug_overlay(frame, mask_rs, alpha=0.4)
            #cv2.imwrite("/tmp/debug_overlay.png", debug_img)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max((cv2.contourArea(c) for c in contours), default=0)
        moviment = any(cv2.contourArea(c) >= args.min_area for c in contours)

        now = time.time()
        if moviment and now - last_trigger_time > TRIGGER_COOLDOWN_SECS:
            log("Mouvement détecté")
            last_trigger_time = now
            beep_stop = Event()
            beep_procs = beep(1, stop_event=beep_stop)
            Thread(target=record_video,
                   args=(url, auth, list(prebuffer_frames)),
                   kwargs={"duration": args.record_duration, "fps": args.fps},
                   daemon=True).start()
            ui_queue.put(("show_video",
                          build_video_data(prebuffer_frames, url, auth, args.fps, args.prebuffer_secs, args.live_secs,
                                         beep_stop, beep_procs)))
        prev = gray
```

Le remplacer ENTIÈREMENT par :

```python
        fgmask = bg_subtractor.apply(frame)
        # MOG2 marque les ombres à 127 ; on ne garde que l'avant-plan franc (255)
        fgmask = (fgmask >= 250).astype(np.uint8) * 255
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        if mask_rs is not None:
            fgmask = cv2.bitwise_and(fgmask, fgmask, mask=mask_rs)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moviment = any(cv2.contourArea(c) >= args.min_area for c in contours)

        now = time.time()
        if (moviment and now - last_motion_check > YOLO_CHECK_COOLDOWN_SECS
                and now - last_trigger_time > TRIGGER_COOLDOWN_SECS):
            last_motion_check = now
            log("Mouvement détecté")
            confirmed, label, conf = confirm_interesting_object(url, auth, frame)
            if confirmed:
                last_trigger_time = now
                log(f"Confirmation YOLO: {label} ({conf:.2f})")
                beep_stop = Event()
                beep_procs = beep(1, stop_event=beep_stop)
                Thread(target=record_video,
                       args=(url, auth, list(prebuffer_frames)),
                       kwargs={"duration": args.record_duration, "fps": args.fps},
                       daemon=True).start()
                ui_queue.put(("show_video",
                              build_video_data(prebuffer_frames, url, auth, args.fps, args.prebuffer_secs, args.live_secs,
                                             beep_stop, beep_procs)))
            else:
                log("Confirmation YOLO: aucune classe interessante")
```

- [ ] **Step 3 : Vérifier l'import (smoke syntaxe)**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 -c "import sys; sys.path.insert(0,'app'); import detect; print('import OK')"`
Expected: `import OK` (aucune SyntaxError, aucune référence à `prev` ou `args.sensitivity` restante).

- [ ] **Step 4 : Vérifier qu'aucune référence morte à `prev`/`gray` ne subsiste**

Run: `cd /home/m/Bureau/camera-venv/app && grep -nE "\bprev\b|GaussianBlur|args\.sensitivity" detect.py`
Expected: aucune ligne dans `detection()` (les seules occurrences éventuelles de `args.sensitivity` sont dans `main()` au parsing CLI, qui reste inchangé).

- [ ] **Step 5 : Relancer la suite de tests (non-régression)**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_detection_confirm.py && ./bin/python3 app/tests/test_corrupt_frame.py`
Expected: `6/6 OK` puis `6/6 OK`.

- [ ] **Step 6 : Commit**

```bash
cd /home/m/Bureau/camera-venv/app
git add detect.py
git commit -m "feat: detection 2 etages (MOG2 + garde de confirmation YOLO)"
```

---

### Task 4 : Vérification manuelle en conditions réelles

**Files:** aucun (vérification).

- [ ] **Step 1 : Lancer la détection et observer le log**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/detect.py`
Laisser tourner. Provoquer un passage réel (personne) devant la caméra puis un faux mouvement (ex. variation de lumière, ombre).

- [ ] **Step 2 : Vérifier le comportement attendu dans le log du jour**

Run (dans un autre terminal) : `grep -E "Mouvement détecté|Confirmation YOLO" /home/m/Bureau/camera-venv/app/logs/$(date +%Y.%m.%d).detect.log | tail -20`
Attendu :
- Sur un vrai passage : `Mouvement détecté` suivi de `Confirmation YOLO: personne (0.xx)` puis enregistrement.
- Sur un faux mouvement : `Mouvement détecté` suivi de `Confirmation YOLO: aucune classe interessante`, sans klaxon/enregistrement.

- [ ] **Step 3 : Arrêter et relancer en service**

Ctrl-C sur l'instance de test. La remise en service se fait par le watchdog cron (ou `./app/restart_detect.sh`). Confirmer le code à jour : `pkill -f app/detect.py` puis `/home/m/Bureau/camera-venv/app/cron_detect.sh`.

---

## Notes de fin

- `--sensitivity` n'a plus d'effet (MOG2 utilise `MOG2_VAR_THRESHOLD`) ; l'argument CLI est conservé pour compatibilité mais ignoré. `--min-area` reste utilisé (surface min des contours d'avant-plan).
- `_compute_yolo_score` / `select_best_frame` (sélection de frame ntfy) restent inchangés.
- Le filtre `is_corrupted_frame` reste en tête de boucle : une frame corrompue ne pollue donc pas non plus le modèle de fond MOG2.
- Réglages à ajuster si besoin après observation : `YOLO_CONFIRM_CONF` (recall vs faux positifs), `MOG2_VAR_THRESHOLD` (sensibilité mouvement), `YOLO_CHECK_COOLDOWN_SECS` (fréquence des vérifications).
- Après validation, fusionner `feature/detection-2-stages` dans `master`.
```
