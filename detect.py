#!/usr/bin/env python3
import os
os.environ.setdefault("DISPLAY", ":0")
import cv2
import numpy as np
import requests
import time
import argparse
import datetime
import subprocess
from threading import Thread, Lock, Event
from collections import deque
from queue import Queue
from rate_mov import score_video_file
import tkinter as tk
from PIL import Image, ImageTk

video_recording = False
video_lock = Lock()
last_trigger_time = 0
last_ntfy_time = 0

ui_queue = Queue()
stop_event = Event()

# ──────────────────────────────────────────────────────────────────────────────
# Settings (placer les paramètres globaux le plus haut possible)
# ──────────────────────────────────────────────────────────────────────────────
# Durée de la vidéo affichée dans la petite fenêtre
# - PREBUFFER_SECS: durée du prébuffer (en secondes) rejouée avant le live
# - LIVE_SECS: durée du live (en secondes) avant fermeture automatique
PREBUFFER_SECS = 3
LIVE_SECS = 10

# Chemins et périphériques
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(APP_DIR, "captures")
MASK_PATH = os.path.join(APP_DIR, "masque.png")
BEEP_PATH = os.path.join(APP_DIR, "klaxon.aac")

# Caméra / flux
CAMERA_URL = "http://192.168.1.111/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"
CAMERA_AUTH = ("admin", "password")
FETCH_TIMEOUT = 5  # s

# Notification ntfy
NTFY_TOPIC = "https://ntfy.sh/your-private-topic"
NTFY_COOLDOWN_SECS = 120  # delai minimum entre deux envois d'images
NTFY_CANDIDATE_SECONDS = [5, 7, 9, 11]  # secondes de capture pour selection intelligente

# YOLO detection pour selection frame ntfy
YOLO_CFG = os.path.join(APP_DIR, "models/yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(APP_DIR, "models/yolov4-tiny.weights")
YOLO_CLASSES = os.path.join(APP_DIR, "models/coco.names")
INTERESTING_CLASSES = {0: "personne", 1: "velo", 2: "voiture", 3: "moto", 5: "bus", 7: "camion", 15: "chat", 16: "chien"}

# Framerate unique (analyse, UI, enregistrement)
FPS_DEFAULT = 10
SENSITIVITY_DEFAULT = 100
MIN_AREA_DEFAULT = 400  # 16/06: 80 trop bas, MOG2 declenchait sur le bruit du capteur

# Détection de frames corrompues (bandes vert/magenta néon de la caméra)
# Calibré : frame corrompue ≈ 0.31 de fraction néon, frames normales = 0.0
CORRUPT_SAT_MIN = 150       # saturation HSV mini pour qu'un pixel compte comme "néon"
CORRUPT_VAL_MIN = 100       # luminosité HSV mini (ignore le bruit sombre)
CORRUPT_SAT_FRAC_MAX = 0.05  # fraction max de pixels néon avant de juger la frame corrompue

# Détection à 2 étages : fond adaptatif (MOG2) + confirmation YOLO
MOG2_HISTORY = 500            # nb de frames pour le modèle de fond
MOG2_VAR_THRESHOLD = 25       # 16 (defaut OpenCV) trop sensible au bruit ; 25 = plus robuste
YOLO_CONFIRM_CONF = 0.55      # 0.4 laissait passer les faux "personne" (ombres du matin a 0.40-0.47)
YOLO_CONFIRM_FRAMES = 3       # nb de frames saines à analyser au plus
YOLO_CONFIRM_INTERVAL = 0.2   # délai (s) entre frames de la rafale
YOLO_CONFIRM_MAX_FETCH = 6    # borne de tentatives (anti-boucle si glitch en rafale)
YOLO_CHECK_COOLDOWN_SECS = 2  # délai mini entre deux confirmations YOLO sur mouvement non confirmé
                              # (compromis : plus court = plus réactif mais plus de CPU YOLO)
MOTION_BOX_MARGIN_PX = 40     # marge autour des contours en mouvement pour le recoupement YOLO
                              # (tolère le déplacement de l'objet pendant la rafale de confirmation)
STATIC_SUPPRESS_AFTER = 2     # nb de confirmations au meme endroit avant de juger l'objet statique
                              # (2 = une seule alerte pour un objet qui arrive puis reste immobile)
STATIC_IOU_MIN = 0.6          # recouvrement mini pour considerer que c'est le meme objet
STATIC_FORGET_SECS = 86400    # sans re-confirmation pendant ce delai (24h), l'objet statique est oublie
STARTUP_WARMUP_SECS = 10      # pas de declenchement au demarrage, le temps que MOG2 apprenne le fond
TRIGGER_COOLDOWN_SECS = 25  # évite les déclenchements trop fréquents

# Enregistrement (durée en secondes)
RECORD_DURATION = 15  # s de live enregistrées en plus du prébuffer
MIN_VIDEO_SIZE_KB = 420
SCORE_MIN = 200000

# Prébuffer mémoire (capacité)
# Capacité par défaut dimensionnée pour couvrir au moins PREBUFFER_SECS à FPS_DEFAULT
PREBUFFER_FRAMES_MAXLEN = max(PREBUFFER_SECS, 10) * FPS_DEFAULT

# Nettoyage
MAX_AGE_HOURS_DEFAULT = 72
CLEAN_INTERVAL_SECONDS = 600

# Son
BEEP_DELAY = 8

# Initialisation dépendante des settings (sera recalculée après parse CLI)
prebuffer_frames = deque(maxlen=PREBUFFER_FRAMES_MAXLEN)

# ──────────────────────────────────────────────────────────────────────────────
# Chargement YOLO pour selection intelligente de frame
# ──────────────────────────────────────────────────────────────────────────────
yolo_net = None
yolo_output_layers = []
yolo_classes = []

def _load_yolo():
    global yolo_net, yolo_output_layers, yolo_classes
    if not os.path.exists(YOLO_CFG) or not os.path.exists(YOLO_WEIGHTS):
        return False
    try:
        yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = yolo_net.getLayerNames()
        yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
        if os.path.exists(YOLO_CLASSES):
            with open(YOLO_CLASSES, 'r') as f:
                yolo_classes = [line.strip() for line in f.readlines()]
        return True
    except Exception as e:
        print(f"Erreur chargement YOLO: {e}")
        return False

_yolo_loaded = _load_yolo()

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def log(message: str) -> None:
    now = datetime.datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    line = f"{timestamp} {message}"
    print(line)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, now.strftime("%Y.%m.%d.detect.log"))

    try:
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

def fetch_image(url: str, auth: tuple[str, str]):
    try:
        r = requests.get(url, auth=auth, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except requests.RequestException as e:
        log(f"Erreur récupération image : {e}")
        return None

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
        green = (h >= 45) & (h <= 85)      # vert néon (OpenCV H 0-179, exclut le cyan)
        magenta = (h >= 140) & (h <= 170)  # magenta
        neon = strong & (green | magenta)
        fraction = float(neon.sum()) / neon.size
        return fraction > CORRUPT_SAT_FRAC_MAX
    except Exception as e:
        log(f"is_corrupted_frame erreur : {e}")
        return False

def beep(n: int, delay: int = BEEP_DELAY,
         path: str = BEEP_PATH, stop_event=None) -> list:
    """Retourne la liste des processus ffplay lancés pour pouvoir les tuer."""
    processes = []
    def _play():
        for _ in range(n):
            if stop_event and stop_event.is_set():
                break
            proc = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", path],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(proc)
            time.sleep(delay)
    Thread(target=_play, daemon=True).start()
    return processes

def debug_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superpose le masque sur l'image réelle avec transparence et écrit YES/NO
    en fonction des zones réellement surveillées.
    """
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[mask == 255] = (0, 255, 0)   # vert = surveillé
    overlay[mask == 0]   = (0, 0, 255)   # rouge = ignoré
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Cherche les contours du masque surveillé
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 500:  # filtre petites zones bruit
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(blended, "YES", (cx-20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)

    # Même principe pour les zones non surveillées (inverser le masque)
    inv_mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(blended, "NO", (cx-20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    return blended


# ──────────────────────────────────────────────────────────────────────────────
# Masque de surveillance
# ──────────────────────────────────────────────────────────────────────────────
def load_surveillance_mask(watched="green"):  # "green" ou "red"
    path = MASK_PATH
    if not os.path.exists(path):
        log(f"Masque non trouvé : {path}")
        return None

    m_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if m_bgr is None:
        log("Erreur : impossible de charger le masque")
        return None

    m_rgb = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)

    if watched == "green":
        sel = np.all(m_rgb == [0, 255, 0], axis=2)
    elif watched == "red":
        sel = np.all(m_rgb == [255, 0, 0], axis=2)
    else:
        raise ValueError("watched must be 'green' or 'red'")

    # 0/255, 1 canal
    mask_u8 = (sel.astype(np.uint8) * 255)
    log(f"Masque chargé {m_rgb.shape[:2]}, zones surveillées (px) : {int(sel.sum())}")
    return mask_u8
# ──────────────────────────────────────────────────────────────────────────────
# Vidéo
# ──────────────────────────────────────────────────────────────────────────────
def _send_ntfy(jpeg_data, timestamp):
    """Envoi bloquant vers ntfy (execute dans un thread separe)."""
    try:
        resp = requests.put(
            NTFY_TOPIC,
            data=jpeg_data,
            headers={
                "Filename": f"capture_{timestamp}.jpg",
                "Title": "Detection mouvement",
                "Priority": "5",
            },
            timeout=10
        )
        if resp.status_code == 200:
            log(f"Ntfy: image publiee ({len(jpeg_data)} octets)")
        else:
            log(f"Ntfy: erreur HTTP {resp.status_code}")
    except Exception as e:
        log(f"Ntfy: echec envoi - {e}")

def _compute_motion_score(frames):
    """Calcule le score de mouvement pour chaque frame (delta avec precedente)."""
    if len(frames) < 2:
        return [0.0] * len(frames)
    scores = [0.0]
    for i in range(1, len(frames)):
        gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(gray_prev, gray_curr)
        scores.append(float(np.sum(delta)))
    max_score = max(scores) if max(scores) > 0 else 1.0
    return [s / max_score for s in scores]

def _compute_yolo_score(frame):
    """Execute YOLO sur une frame et retourne le score max des classes interessantes."""
    if yolo_net is None:
        return 0.0
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_output_layers)
        best_conf = 0.0
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id in INTERESTING_CLASSES and confidence > best_conf:
                    best_conf = float(confidence)
        return best_conf
    except Exception as e:
        log(f"Ntfy: erreur YOLO - {e}")
        return 0.0

def is_interesting_detection(class_id, confidence, threshold=YOLO_CONFIRM_CONF):
    """Règle de décision : vrai si la classe est surveillée ET la confiance suffisante."""
    return class_id in INTERESTING_CLASSES and confidence >= threshold

def _motion_boxes(contours, min_area, w, h):
    """Boîtes englobantes (x, y, w, h) des contours d'aire >= min_area,
    élargies de MOTION_BOX_MARGIN_PX et clampées aux dimensions de la frame."""
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x0 = max(0, x - MOTION_BOX_MARGIN_PX)
        y0 = max(0, y - MOTION_BOX_MARGIN_PX)
        x1 = min(w, x + bw + MOTION_BOX_MARGIN_PX)
        y1 = min(h, y + bh + MOTION_BOX_MARGIN_PX)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def _overlaps_any(box, boxes):
    """Vrai si le rectangle (x, y, w, h) chevauche au moins un rectangle de boxes."""
    x, y, bw, bh = box
    for mx, my, mw, mh in boxes:
        if x < mx + mw and mx < x + bw and y < my + mh and my < y + bh:
            return True
    return False


def _warmup_active(started_at, now):
    """Vrai pendant la periode de chauffe qui suit le demarrage de la surveillance."""
    return now - started_at < STARTUP_WARMUP_SECS


def _iou(a, b):
    """Intersection sur union de deux rectangles (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# Objets confirmes par YOLO mais immobiles (ex : voiture garee dont les reflets
# declenchent le detecteur de mouvement). Rempli/consulte par _static_check.
static_objects = []


def _static_check(registry, class_id, box, now):
    """Enregistre une confirmation YOLO et renvoie True si l'objet doit etre
    supprime : meme classe confirmee STATIC_SUPPRESS_AFTER fois ou plus au meme
    endroit (IoU >= STATIC_IOU_MIN). Les entrees sans re-confirmation depuis
    STATIC_FORGET_SECS sont oubliees (l'objet redevient signalable)."""
    registry[:] = [e for e in registry if now - e["last_seen"] <= STATIC_FORGET_SECS]
    for e in registry:
        if e["class_id"] == class_id and _iou(e["box"], box) >= STATIC_IOU_MIN:
            e["count"] += 1
            e["last_seen"] = now
            return e["count"] >= STATIC_SUPPRESS_AFTER
    registry.append({"class_id": class_id, "box": box, "count": 1, "last_seen": now})
    return False


def _best_interesting(outputs, w, h, mask=None, motion_boxes=None):
    """Meilleure (class_id, confiance) parmi les classes surveillées, (None, 0.0) sinon.
    - Confiance = objectness * proba_classe (confiance YOLO standard).
    - Si un masque est fourni, le centre de la boîte doit tomber dans une zone
      surveillée (mask[cy, cx] != 0) ; sinon la détection est ignorée.
    - Si motion_boxes est fourni, la boîte YOLO doit chevaucher au moins une
      zone en mouvement ; sinon la détection est ignorée (objet statique).
    - Renvoie aussi la boîte pixels (x, y, w, h) de la détection retenue (ou None)."""
    best_conf = 0.0
    best_id = None
    best_box = None
    for output in outputs:
        for detection in output:
            objectness = float(detection[4])
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = objectness * float(scores[class_id])
            if class_id not in INTERESTING_CLASSES or confidence <= best_conf:
                continue
            if mask is not None:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                if cx < 0 or cy < 0 or cx >= w or cy >= h or mask[cy, cx] == 0:
                    continue
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)
            bx = int(detection[0] * w) - bw // 2
            by = int(detection[1] * h) - bh // 2
            if motion_boxes is not None and not _overlaps_any((bx, by, bw, bh), motion_boxes):
                continue
            best_conf = confidence
            best_id = class_id
            best_box = (bx, by, bw, bh)
    return best_id, best_conf, best_box


def _detect_interesting(frame, mask=None, motion_boxes=None):
    """Renvoie (class_id, confiance) de la meilleure détection d'une classe surveillée
    dans la zone du masque, (None, 0.0) sinon.
    Ne capture PAS les exceptions : elles remontent pour le fail-safe."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(yolo_output_layers)
    return _best_interesting(outputs, w, h, mask, motion_boxes)


def confirm_interesting_object(url, auth, first_frame, mask=None, motion_boxes=None):
    """Confirme la présence d'une classe surveillée via YOLO sur une rafale de frames.
    Renvoie (confirmé: bool, label: str|None, confiance: float).
    - Analyse d'abord first_frame (déjà filtrée), puis récupère des frames live.
    - Ignore les frames corrompues / None (ne comptent pas, on en refetch une autre).
    - Restreint les détections à la zone du masque (objet hors zone = ignoré).
    - Restreint aux détections chevauchant une zone en mouvement (objet statique = ignoré).
    - Supprime les objets confirmés en boucle au même endroit (voir _static_check).
    - Fail-safe : YOLO indisponible ou inférence en erreur -> (True, None, 0.0)."""
    if yolo_net is None:
        return True, None, 0.0
    frame = first_frame
    analysed = 0
    static_logged = False
    try:
        for _ in range(YOLO_CONFIRM_MAX_FETCH):
            if frame is not None and not is_corrupted_frame(frame):
                class_id, conf, box = _detect_interesting(frame, mask, motion_boxes)
                if class_id is not None and is_interesting_detection(class_id, conf):
                    if _static_check(static_objects, class_id, box, time.time()):
                        if not static_logged:
                            log(f"Confirmation YOLO: {INTERESTING_CLASSES[class_id]} statique, ignore")
                            static_logged = True
                    else:
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

def select_best_frame(candidates):
    """Selectionne la meilleure frame parmi les candidates (score hybride mouvement+YOLO)."""
    if not candidates:
        return None, -1
    n = len(candidates)
    log(f"Ntfy: analyse {n} candidates")

    frames = [c[1] for c in candidates]
    motion_scores = _compute_motion_score(frames)

    yolo_scores = []
    for frame in frames:
        yolo_scores.append(_compute_yolo_score(frame))

    best_idx = 0
    best_score = -1.0
    for i in range(n):
        composite = 0.4 * motion_scores[i] + 0.6 * yolo_scores[i]
        if composite > best_score:
            best_score = composite
            best_idx = i

    sec = candidates[best_idx][0]
    log(f"Ntfy: meilleure frame sec {sec} (score {best_score:.3f}, motion={motion_scores[best_idx]:.3f}, yolo={yolo_scores[best_idx]:.3f})")
    return candidates[best_idx][1], best_idx

def _publish_best_frame_thread(candidates, timestamp):
    """Thread d'analyse et envoi de la meilleure frame."""
    best_frame, _ = select_best_frame(candidates)
    if best_frame is None:
        log("Ntfy: aucune frame candidate")
        return
    try:
        _, jpeg = cv2.imencode('.jpg', best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_data = jpeg.tobytes()
        _send_ntfy(jpeg_data, timestamp)
    except Exception as e:
        log(f"Ntfy: erreur encodage - {e}")

def publish_best_frame(candidates):
    """Lance l'analyse et l'envoi dans un thread daemon (non-bloquant)."""
    global last_ntfy_time
    now = time.time()
    if now - last_ntfy_time < NTFY_COOLDOWN_SECS:
        return
    hour = datetime.datetime.now().hour
    if hour >= 20 or hour < 8:
        return
    last_ntfy_time = now
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    t = Thread(target=_publish_best_frame_thread, args=(candidates, ts), daemon=True)
    t.start()

def publish_to_ntfy(frame):
    """Encode et publie une frame vers ntfy dans un thread daemon (non-bloquant)."""
    global last_ntfy_time
    now = time.time()
    if now - last_ntfy_time < NTFY_COOLDOWN_SECS:
        return
    hour = datetime.datetime.now().hour
    if hour >= 20 or hour < 8:
        return
    last_ntfy_time = now
    try:
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_data = jpeg.tobytes()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        t = Thread(target=_send_ntfy, args=(jpeg_data, ts), daemon=True)
        t.start()
    except Exception as e:
        log(f"Ntfy: erreur encodage - {e}")

def record_video(url, auth, prebuffer, duration=RECORD_DURATION, fps=FPS_DEFAULT,
                 output_dir=CAPTURES_DIR):
    """Enregistre le prebuffer puis 30 s d’images."""
    global video_recording
    with video_lock:
        if video_recording:
            return
        video_recording = True

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    filename = os.path.join(output_dir, f"descente_{ts}.avi")

    if not prebuffer:
        log("Buffer préenregistrement vide")
        video_recording = False
        return

    h, w, _ = prebuffer[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    for f in prebuffer:
        out.write(f)

    start_live = time.time()
    end = start_live + duration
    ntfy_candidates = []
    captured_seconds = set()
    while time.time() < end:
        frame = fetch_image(url, auth)
        if frame is not None:
            out.write(frame)
            elapsed = time.time() - start_live
            current_sec = int(elapsed)
            if current_sec in NTFY_CANDIDATE_SECONDS and current_sec not in captured_seconds:
                ntfy_candidates.append((current_sec, frame.copy()))
                captured_seconds.add(current_sec)
        time.sleep(1 / fps)
    out.release()

    if ntfy_candidates:
        publish_best_frame(ntfy_candidates)

    try:
        size_kb = os.path.getsize(filename) / 1024
        if size_kb < MIN_VIDEO_SIZE_KB:
            os.remove(filename)
            log(f"Vidéo supprimée : {filename} (taille {int(size_kb)} Ko)")
        else:
            score = score_video_file(filename)
            if score < SCORE_MIN:
                os.remove(filename)
                log(f"Vidéo supprimée : {filename} (score {score})")
            else:
                log(f"Vidéo enregistrée : {filename} (score {score})")
    except Exception as e:
        log(f"Post-enregistrement erreur : {e}")

    video_recording = False

def clean_old_videos(folder=CAPTURES_DIR,
                     max_age_hours=MAX_AGE_HOURS_DEFAULT, interval_seconds=CLEAN_INTERVAL_SECONDS):
    while not stop_event.is_set():
        now = datetime.datetime.now()
        for f in os.listdir(folder):
            if f.startswith("descente_") and f.endswith(".avi"):
                try:
                    dt = datetime.datetime.strptime(f[9:-4], "%Y_%m_%d-%H-%M-%S")
                    if (now - dt).total_seconds() / 3600 > max_age_hours:
                        os.remove(os.path.join(folder, f))
                        log(f"Supprimé (ancien) : {f}")
                except Exception as e:
                    log(f"Analyse fichier {f} : {e}")
        time.sleep(interval_seconds)

# ──────────────────────────────────────────────────────────────────────────────
# Interface : lecture prebuffer + live
# ──────────────────────────────────────────────────────────────────────────────
def build_video_data(prebuffer, url, auth, fps=FPS_DEFAULT,
                     pre_secs=PREBUFFER_SECS, live_secs=LIVE_SECS,
                     beep_stop_event=None, beep_processes=None):
    # On garde seulement les N secondes les plus récentes du prebuffer
    pre_frames = list(prebuffer)[-pre_secs*fps:]
    return {
        "pre_frames": pre_frames,
        "url": url,
        "auth": auth,
        "fps": fps,
        "live_secs": live_secs,
        "beep_stop_event": beep_stop_event,
        "beep_processes": beep_processes or []
    }

def show_buffer_video(parent, data):
    pre_frames = data["pre_frames"]
    url, auth = data["url"], data["auth"]
    fps, live_secs = data["fps"], data["live_secs"]
    beep_stop_event = data.get("beep_stop_event")
    beep_processes = data.get("beep_processes", [])

    # Sauvegarder la fenetre active avant d'afficher
    try:
        prev_win_id = subprocess.check_output(["xdotool", "getactivewindow"]).strip()
    except Exception:
        prev_win_id = None

    win = tk.Toplevel(parent)
    win.title("Mouvement détecté")
    h, w, _ = pre_frames[0].shape
    win.geometry(f"{w}x{h}+100+100")
    win.attributes("-alpha", 0.7)
    win.attributes("-topmost", True)

    def _restore_focus():
        win.attributes("-topmost", False)
        win.attributes("-alpha", 0.7)
        if prev_win_id:
            try:
                subprocess.Popen(["xdotool", "windowactivate", prev_win_id],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    win.after(200, _restore_focus)

    panel = tk.Label(win)
    panel.pack()

    idx = 0
    live_start = None

    def close(_=None):
        # Arrête le klaxon
        if beep_stop_event:
            beep_stop_event.set()
        for proc in beep_processes:
            try:
                proc.terminate()
            except:
                pass
        if win.winfo_exists():
            win.destroy()

    win.bind("<Button-1>", close)
    panel.bind("<Button-1>", close)

    def update():
        nonlocal idx, live_start
        if not win.winfo_exists():
            return

        if idx < len(pre_frames):
            frame = pre_frames[idx]
            idx += 1
        else:
            if live_start is None:
                live_start = time.time()
            elif time.time() - live_start >= live_secs:
                close()
                return
            frame = fetch_image(url, auth)
            if frame is None:
                win.after(int(1000 / fps), update)
                return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel.configure(image=img)
        panel.image = img
        win.after(int(1000 / fps), update)

    win.after(0, update)

# ──────────────────────────────────────────────────────────────────────────────
# Détection
# ──────────────────────────────────────────────────────────────────────────────
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

    # Laisse la CLI overrider la rétention via --max-age-hours
    Thread(target=clean_old_videos, args=(CAPTURES_DIR, args.max_age_hours, CLEAN_INTERVAL_SECONDS), daemon=True).start()
    log("Surveillance démarrée")
    warmup_started = time.time()

    while not stop_event.is_set():
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(delay)
            continue

        if is_corrupted_frame(frame):
            log("Frame corrompue ignorée")
            time.sleep(delay)
            continue

        prebuffer_frames.append(frame.copy())

        if mask is not None and mask_rs is None:
            mask_rs = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            # S’assure que c’est bien 1 canal uint8 0/255
            if mask_rs.ndim == 3:
                mask_rs = cv2.cvtColor(mask_rs, cv2.COLOR_BGR2GRAY)
            mask_rs = (mask_rs > 127).astype(np.uint8) * 255
            log(f"Masque redimensionné : {mask.shape} → {mask_rs.shape}")

        fgmask = bg_subtractor.apply(frame)
        # MOG2 marque les ombres à 127 ; on ne garde que l'avant-plan franc (255)
        fgmask = (fgmask >= 250).astype(np.uint8) * 255
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        if mask_rs is not None:
            fgmask = cv2.bitwise_and(fgmask, fgmask, mask=mask_rs)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_boxes = _motion_boxes(contours, args.min_area, frame.shape[1], frame.shape[0])
        moviment = bool(motion_boxes)

        now = time.time()
        # last_motion_check : rate-limite les vérifications YOLO sur mouvement non confirmé.
        # last_trigger_time : cooldown long après un déclenchement réellement confirmé.
        if (moviment and not _warmup_active(warmup_started, now)
                and now - last_motion_check > YOLO_CHECK_COOLDOWN_SECS
                and now - last_trigger_time > TRIGGER_COOLDOWN_SECS):
            last_motion_check = now
            log("Mouvement détecté")
            confirmed, label, conf = confirm_interesting_object(url, auth, frame, mask_rs,
                                                                motion_boxes)
            if confirmed:
                last_trigger_time = now
                log(f"Confirmation YOLO: {label or 'fail-open'} ({conf:.2f})")
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
        time.sleep(delay)

# ──────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Détection de mouvement caméra IP")
    p.add_argument("--fps", type=int, default=FPS_DEFAULT, help="FPS global (analyse, UI, enregistrement)")
    p.add_argument("--prebuffer-secs", type=int, default=PREBUFFER_SECS, help="Durée du prébuffer (s)")
    p.add_argument("--live-secs", type=int, default=LIVE_SECS, help="Durée du live (s)")
    p.add_argument("--record-duration", type=int, default=RECORD_DURATION, help="Durée d'enregistrement live (s)")
    p.add_argument("--sensitivity", type=int, default=SENSITIVITY_DEFAULT, help="Ignoré (MOG2 utilise MOG2_VAR_THRESHOLD) - conservé pour compat CLI")
    p.add_argument("--min-area", type=int, default=MIN_AREA_DEFAULT, help="Surface min px²")
    p.add_argument("--max-age-hours", type=int, default=MAX_AGE_HOURS_DEFAULT, help="Rétention vidéos")
    p.add_argument("--test", action="store_true", help="Simule un mouvement toutes les 40s")
    args = p.parse_args()

    # Recalcule la capacité du prébuffer d'après les choix CLI
    global prebuffer_frames
    prebuffer_frames = deque(maxlen=max(args.prebuffer_secs, 10) * args.fps)

    root = tk.Tk(); root.withdraw()

    def poll_queue():
        try:
            while True:
                action, data = ui_queue.get_nowait()
                if action == "show_video":
                    show_buffer_video(root, data)
                ui_queue.task_done()
        except:
            pass
        if root.winfo_exists():
            root.after(100, poll_queue)

    Thread(target=detection, args=(args,), daemon=True).start()

    if args.test:
        def fake_trigger():
            time.sleep(15)
            if stop_event.is_set():
                return
            log("TEST: simulation mouvement")
            beep_stop = Event()
            beep_procs = beep(1, stop_event=beep_stop)
            if prebuffer_frames:
                pass
            else:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                for _ in range(args.fps * args.prebuffer_secs):
                    prebuffer_frames.append(dummy)
            ui_queue.put(("show_video",
                          build_video_data(prebuffer_frames, url, auth, args.fps, args.prebuffer_secs, args.live_secs,
                                           beep_stop, beep_procs)))
            while not stop_event.is_set():
                time.sleep(40)
                if stop_event.is_set():
                    break
                log("TEST: simulation mouvement")
                beep_stop = Event()
                beep_procs = beep(1, stop_event=beep_stop)
                # Genere des frames de test (gris) si le prebuffer est vide
                if prebuffer_frames:
                    frames = list(prebuffer_frames)
                else:
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                    frames = [dummy] * (args.fps * args.prebuffer_secs)
                    for f in frames:
                        prebuffer_frames.append(f)
                ui_queue.put(("show_video",
                              build_video_data(prebuffer_frames, url, auth, args.fps, args.prebuffer_secs, args.live_secs,
                                               beep_stop, beep_procs)))
        url, auth = CAMERA_URL, CAMERA_AUTH
        Thread(target=fake_trigger, daemon=True).start()

    poll_queue()

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nArrêt utilisateur")
