# Filtre de frames corrompues — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empêcher les frames corrompues de la caméra (bandes vert/magenta néon) de déclencher de faux mouvements, en les détectant et en les ignorant dans la boucle de détection de `app/detect.py`.

**Architecture:** Une fonction pure `is_corrupted_frame(frame)` mesure la fraction de pixels fortement saturés en teinte vert néon ou magenta (signature du défaut). Dans `detection()`, juste après la récupération de la frame, une frame jugée corrompue est ignorée (skip) avant d'entrer dans le prébuffer, de devenir `prev` ou de déclencher quoi que ce soit. Le prébuffer reste propre automatiquement puisqu'il est rempli dans cette même boucle.

**Tech Stack:** Python 3.13, OpenCV (`cv2`), NumPy. Venv à la racine (`./bin/python3`). Pas de framework de test : tests sous forme de script autonome exécuté avec l'interpréteur du venv.

**Seuils (calibrés empiriquement) :** frame corrompue ≈ 0,31 de fraction néon ; frames normales = 0,0000. Seuil retenu : 0,05 (grosse marge des deux côtés).

**Préliminaire :** Travailler sur une branche dédiée.
```bash
cd /home/m/Bureau/camera-venv/app
git checkout -b feature/corrupt-frame-gate
```

---

### Task 1 : Fonction `is_corrupted_frame` + constantes

**Files:**
- Create: `app/tests/test_corrupt_frame.py`
- Modify: `app/detect.py` (ajout des constantes vers la ligne 60, ajout de la fonction dans la section « Utilitaires » vers la ligne 150)

- [ ] **Step 1 : Écrire le test qui échoue**

Créer `app/tests/test_corrupt_frame.py` :

```python
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
```

- [ ] **Step 2 : Lancer le test pour vérifier qu'il échoue**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_corrupt_frame.py`
Expected: FAIL/ERROR sur tous les tests, `AttributeError: module 'detect' has no attribute 'is_corrupted_frame'`.

- [ ] **Step 3 : Ajouter les constantes dans `detect.py`**

Dans la section Settings, juste après la ligne `MIN_AREA_DEFAULT = 80` (~ligne 60), ajouter :

```python
# Détection de frames corrompues (bandes vert/magenta néon de la caméra)
# Calibré : frame corrompue ≈ 0.31 de fraction néon, frames normales = 0.0
CORRUPT_SAT_MIN = 150       # saturation HSV mini pour qu'un pixel compte comme "néon"
CORRUPT_VAL_MIN = 100       # luminosité HSV mini (ignore le bruit sombre)
CORRUPT_SAT_FRAC_MAX = 0.05  # fraction max de pixels néon avant de juger la frame corrompue
```

- [ ] **Step 4 : Implémenter `is_corrupted_frame` dans `detect.py`**

Dans la section « Utilitaires », après la fonction `fetch_image` (~ligne 135), ajouter :

```python
def is_corrupted_frame(frame) -> bool:
    """Vrai si la frame présente la signature de corruption caméra
    (forte proportion de pixels saturés en vert néon ou magenta).
    Fail-open : retourne False en cas d'erreur pour ne jamais aveugler la caméra.
    """
    try:
        if frame is None or frame.size == 0:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        strong = (s >= CORRUPT_SAT_MIN) & (v >= CORRUPT_VAL_MIN)
        green = (h >= 45) & (h <= 90)      # vert néon (OpenCV H 0-179)
        magenta = (h >= 140) & (h <= 170)  # magenta
        neon = strong & (green | magenta)
        fraction = float(neon.sum()) / neon.size
        return fraction > CORRUPT_SAT_FRAC_MAX
    except Exception as e:
        log(f"is_corrupted_frame erreur : {e}")
        return False
```

- [ ] **Step 5 : Lancer le test pour vérifier qu'il passe**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_corrupt_frame.py`
Expected: `6/6 OK`, code de sortie 0.

- [ ] **Step 6 : Commit**

```bash
cd /home/m/Bureau/camera-venv/app
git add detect.py tests/test_corrupt_frame.py
git commit -m "feat: detection de frames corrompues (is_corrupted_frame)"
```

---

### Task 2 : Brancher le filtre dans la boucle `detection()`

**Files:**
- Modify: `app/detect.py` (fonction `detection`, juste après le `fetch_image` / check `None`, ~ligne 537-540)

- [ ] **Step 1 : Ajouter le skip dans la boucle**

Dans `detection()`, le code actuel est :

```python
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(delay)
            continue

        prebuffer_frames.append(frame.copy())
```

Insérer le filtre **entre** le check `None` et `prebuffer_frames.append` :

```python
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(delay)
            continue

        if is_corrupted_frame(frame):
            log("Frame corrompue ignorée")
            time.sleep(delay)
            continue

        prebuffer_frames.append(frame.copy())
```

- [ ] **Step 2 : Vérifier que le module s'importe sans erreur (smoke test syntaxe)**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 -c "import sys; sys.path.insert(0,'app'); import detect; print('import OK')"`
Expected: `import OK` (aucune SyntaxError).

- [ ] **Step 3 : Relancer la suite de tests (non-régression)**

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/tests/test_corrupt_frame.py`
Expected: `6/6 OK`.

- [ ] **Step 4 : Commit**

```bash
cd /home/m/Bureau/camera-venv/app
git add detect.py
git commit -m "feat: ignore les frames corrompues dans la boucle de detection"
```

---

### Task 3 : Vérification manuelle sur l'échantillon réel

**Files:** aucun (vérification).

- [ ] **Step 1 : Confirmer la détection sur la vraie frame corrompue**

Le fichier `artifacts_tmp/2026-05-28_11-41.png` est hors du dépôt (racine `camera-venv/`). Vérifier qu'il déclenche bien la détection (et qu'une frame normale d'une capture ne la déclenche pas) :

Run:
```bash
cd /home/m/Bureau/camera-venv && ./bin/python3 -c "
import sys, cv2, glob; sys.path.insert(0,'app'); import detect
img = cv2.imread('artifacts_tmp/2026-05-28_11-41.png')[28:545, 3:643]
print('corrompue ->', detect.is_corrupted_frame(img))
vids = sorted(glob.glob('app/captures/*.avi'))
if vids:
    cap = cv2.VideoCapture(vids[0]); ok, fr = cap.read(); cap.release()
    if ok: print('normale   ->', detect.is_corrupted_frame(fr))
"
```
Expected: `corrompue -> True` et (si une capture existe) `normale -> False`.

- [ ] **Step 2 : (Optionnel) Test live court**

Lancer la détection quelques minutes en heure de glitch connue et vérifier l'apparition de « Frame corrompue ignorée » dans `app/logs/AAAA.MM.JJ.detect.log`, sans perte de détection des vrais mouvements.

Run: `cd /home/m/Bureau/camera-venv && ./bin/python3 app/detect.py`
(Ctrl-C pour arrêter ; puis `grep "Frame corrompue" app/logs/$(date +%Y.%m.%d).detect.log`.)

---

## Notes de fin

- Le seuil `CORRUPT_SAT_FRAC_MAX = 0.05` est conservateur (normales à 0,0). S'il fallait l'ajuster, c'est la seule valeur à toucher.
- Aucune modification de `record_video`, `select_best_frame` ou ntfy : périmètre limité à la boucle de détection (décision validée en spec).
- Après validation, fusionner `feature/corrupt-frame-gate` dans `master`.
