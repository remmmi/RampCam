import cv2
import os
import re
import time
import numpy as np
from statistics import stdev
from collections import deque
import urllib.request
import pathlib

motion_tests = []

def register_test(weight_range=(0, 10), pow=1.0):
    def decorator(func):
        func._weight_range = weight_range
        func._pow = pow
        motion_tests.append(func)
        return func
    return decorator

def apply_pow(pow=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result ** pow
        return wrapper
    return decorator

def strip_score_prefix(filename):
    return re.sub(r"^\d+_", "", filename)

def extract_frames(video_path, max_frames=300):
    """
    Extrait à la fois les frames en couleur et en niveaux de gris d'une vidéo
    """
    cap = cv2.VideoCapture(video_path)
    gray_frames = []
    color_frames = []
    
    if not cap.isOpened():
        return [], []
        
    while True:
        ret, frame = cap.read()
        if not ret or len(gray_frames) >= max_frames:
            break
        color_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        
    cap.release()
    return gray_frames, color_frames

# Chemins pour les modèles YOLO
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Classes que nous voulons détecter (COCO dataset)
INTERESTING_CLASSES = {
    0: "personne",    # person
    1: "vélo",        # bicycle
    2: "voiture",     # car
    3: "moto",        # motorcycle
    5: "bus",         # bus
    6: "train",       # train
    7: "camion",      # truck
    9: "feu_traffic", # traffic light
    15: "chat",       # cat 
    16: "chien",      # dog
    17: "cheval",     # horse
    18: "mouton",     # sheep
    19: "vache",      # cow
    20: "éléphant",   # elephant
    21: "ours",       # bear
    22: "zèbre",      # zebra
    23: "girafe"      # giraffe
}

def download_yolo_models():
    """
    Télécharge les modèles YOLOv4-tiny s'ils n'existent pas encore
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Fichiers de configuration et poids pour YOLOv4-tiny
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    classes_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    
    files = {
        "yolov4-tiny.cfg": config_url,
        "yolov4-tiny.weights": weights_url,
        "coco.names": classes_url
    }
    
    for file_name, url in files.items():
        file_path = os.path.join(MODELS_DIR, file_name)
        if not os.path.exists(file_path):
            try:
                print(f"Téléchargement de {file_name}...")
                urllib.request.urlretrieve(url, file_path)
                print(f"Téléchargé: {file_path}")
            except Exception as e:
                print(f"Erreur téléchargement {file_name}: {e}")
                # Si téléchargement échoue, on essaie de continuer quand même

def initialize_detectors():
    """
    Initialise les détecteurs pour les humains, animaux et véhicules
    """
    detectors = {}
    
    # Détecteur de visages (Haar Cascade)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detectors['face'] = cv2.CascadeClassifier(face_cascade_path)
    
    # Détecteur de corps
    body_cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    detectors['body'] = cv2.CascadeClassifier(body_cascade_path)
    
    # Détecteur HOG pour humains
    detectors['hog'] = cv2.HOGDescriptor()
    detectors['hog'].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Préparation pour YOLO
    try:
        download_yolo_models()
        
        # Chargement du modèle YOLO
        config_path = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
        weights_path = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
        
        if os.path.exists(config_path) and os.path.exists(weights_path):
            detectors['yolo'] = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            # Utiliser CPU ou GPU selon disponibilité
            detectors['yolo'].setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            detectors['yolo'].setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Charger les noms des classes
            classes_path = os.path.join(MODELS_DIR, "coco.names")
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    detectors['classes'] = f.read().strip().split('\n')
            else:
                print("Fichier de classes COCO non trouvé")
        else:
            print("Modèle YOLO non disponible")
    except Exception as e:
        print(f"Erreur initialisation YOLO: {e}")
    
    return detectors

# Background subtractor plus sophistiqué
def create_bg_subtractor():
    """
    Crée un extracteur de fond (background subtractor) sophistiqué
    MOG2 est plus sensible aux petits mouvements et gère mieux les changements de luminosité
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=500,     # Nombre de frames utilisées pour le modèle de fond
        varThreshold=16, # Seuil de variance pour décider si un pixel est avant-plan
        detectShadows=True  # Détection des ombres pour les distinguer des objets mobiles
    )

# Détection d'objets avec YOLO
def detect_objects_yolo(frame, detectors, conf_threshold=0.5, nms_threshold=0.4):
    """
    Détecte des objets dans une image en utilisant YOLO
    Retourne une liste de tuples (classe, confiance, boite)
    """
    if 'yolo' not in detectors or 'classes' not in detectors:
        return []
        
    height, width = frame.shape[:2]
    
    # Transformer l'image pour YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    detectors['yolo'].setInput(blob)
    
    # Obtenir les couches de sortie
    output_layer_names = detectors['yolo'].getUnconnectedOutLayersNames()
    layer_outputs = detectors['yolo'].forward(output_layer_names)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Traiter chaque détection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold and class_id in INTERESTING_CLASSES:
                # Coordonnées du rectangle de détection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculer le coin supérieur gauche
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Appliquer la suppression des non-maximums
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            class_id = class_ids[i]
            if class_id in INTERESTING_CLASSES:
                detections.append((INTERESTING_CLASSES[class_id], confidences[i], boxes[i]))
                
    return detections

@register_test(weight_range=(0, 20), pow=1.5)
def test_detection_multiclasse(frames_data):
    """
    Détecte plusieurs classes d'objets intéressants : personnes, animaux, véhicules
    """
    gray_frames, color_frames = frames_data
    if not color_frames or len(color_frames) < 3:
        return 0.0
        
    detectors = initialize_detectors()
    if 'yolo' not in detectors:
        # Si YOLO n'est pas disponible, on retourne 0
        print("YOLO non disponible pour détection multi-classe")
        return 0.0
        
    # Résultats de détection
    total_confidence = 0.0
    max_confidence_by_class = {}
    detected_frames = 0
    
    # Nombre de frames à analyser (on prend 1 frame sur 3 pour optimiser)
    sample_frames = color_frames[::2]  
    
    for frame in sample_frames:
        # Détecter les objets avec YOLO
        detections = detect_objects_yolo(frame, detectors, conf_threshold=0.4)
        
        if detections:
            detected_frames += 1
            frame_confidence = 0.0
            
            # Pour cette frame, on garde la meilleure confiance pour chaque classe
            frame_max_by_class = {}
            
            for obj_class, confidence, box in detections:
                if obj_class not in frame_max_by_class or confidence > frame_max_by_class[obj_class]:
                    frame_max_by_class[obj_class] = confidence
                    
            # Ajouter les scores par classe de cette frame
            for obj_class, conf in frame_max_by_class.items():
                if obj_class not in max_confidence_by_class or conf > max_confidence_by_class[obj_class]:
                    max_confidence_by_class[obj_class] = conf
                    
            # Le score de cette frame est la somme des confiances
            frame_confidence = sum(frame_max_by_class.values()) / len(frame_max_by_class)
            total_confidence += frame_confidence
    
    if detected_frames == 0:
        return 0.0
    
    # Calculer le score final
    # Nous favorisons la détection de plusieurs objets différents
    avg_confidence = total_confidence / detected_frames
    class_diversity = min(1.0, len(max_confidence_by_class) / 3.0)  # 3+ classes = score maximal
    
    # Log des détections pour débogage
    detection_summary = ", ".join([f"{cls}:{conf:.2f}" for cls, conf in max_confidence_by_class.items()])
    if max_confidence_by_class:
        print(f"Détections: {detection_summary}")
    
    return min(1.0, (avg_confidence * 0.6) + (class_diversity * 0.4))

# Nouveaux tests avancés

@register_test(weight_range=(0, 15), pow=1.2)
def test_detection_humaine(frames_data):
    """
    Détecte des personnes en utilisant HOG et Haar cascades
    """
    gray_frames, color_frames = frames_data
    if not color_frames or len(color_frames) < 5:
        return 0.0
        
    detectors = initialize_detectors()
    detected_frames = 0
    total_confidence = 0
    
    # On ne teste pas toutes les frames pour optimiser
    sample_frames = color_frames[::3]  # Prend 1 frame sur 3
    
    for frame in sample_frames:
        # Détection avec HOG (plus fiable pour les personnes à distance)
        rectangles, weights = detectors['hog'].detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4), 
            scale=1.05
        )
        
        # Détection de visages (utile pour les personnes proches)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectors['face'].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(rectangles) > 0 or len(faces) > 0:
            detected_frames += 1
            # On calcule un score de confiance basé sur le nombre et la confiance des détections
            confidence = min(1.0, (len(rectangles) + len(faces)) / 3)
            if len(weights) > 0:
                confidence += min(1.0, np.mean(weights))
            total_confidence += confidence
    
    if detected_frames == 0:
        return 0.0
        
    return min(1.0, (detected_frames / len(sample_frames)) * (total_confidence / detected_frames))

@register_test(weight_range=(0, 12), pow=1.3)
def test_background_subtraction(frames_data):
    """
    Utilise un extracteur de fond MOG2 pour détecter les mouvements importants
    Plus sophistiqué que la différence d'images
    """
    gray_frames, _ = frames_data
    if len(gray_frames) < 10:
        return 0.0
        
    bg_subtractor = create_bg_subtractor()
    movement_scores = []
    
    # Premières frames pour initialiser le modèle de fond
    for i in range(min(10, len(gray_frames))):
        bg_subtractor.apply(gray_frames[i])
    
    # Analyse des frames suivantes
    for i in range(10, len(gray_frames)):
        fg_mask = bg_subtractor.apply(gray_frames[i])
        
        # Application d'opérations morphologiques pour nettoyer le masque
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calcul du pourcentage de pixels en mouvement
        movement_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])
        movement_scores.append(movement_ratio)
    
    if not movement_scores:
        return 0.0
    
    # On calcule à la fois l'intensité moyenne et la cohérence temporelle
    avg_movement = np.mean(movement_scores)
    movement_std = np.std(movement_scores) if len(movement_scores) > 1 else 0
    
    # Un bon mouvement est à la fois significatif et cohérent dans le temps
    coherence_score = max(0, 1.0 - (movement_std * 10))
    movement_score = min(1.0, avg_movement * 20)  # Ajustement pour avoir une échelle appropriée
    
    return min(1.0, (movement_score * 0.7) + (coherence_score * 0.3))

@register_test(weight_range=(0, 10), pow=1.1)
def test_optical_flow(frames_data):
    """
    Utilise le flux optique pour détecter le mouvement directionnel cohérent
    Bon pour identifier les déplacements intentionnels vs aléatoires
    """
    gray_frames, _ = frames_data
    if len(gray_frames) < 10:
        return 0.0
    
    # Paramètres pour ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Paramètres pour Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Sélection des points à suivre dans la première frame
    old_frame = gray_frames[0]
    p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
    if p0 is None:
        return 0.0
    
    # Stockage des directions du mouvement
    flow_directions = []
    coherent_movements = 0
    significant_movements = 0
    
    for i in range(1, len(gray_frames), 3):  # Analyse une frame sur trois
        frame = gray_frames[i]
        if p0 is None or len(p0) < 5:
            p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
            if p0 is None:
                continue
        
        # Calcul du flux optique
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
        
        if p1 is None:
            old_frame = frame
            continue
            
        # Sélection des bons points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Calcul des vecteurs de flux
        if len(good_new) > 0 and len(good_old) > 0:
            flow_vectors = good_new - good_old
            
            # Calcul des directions et magnitudes
            magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
            angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0]) * 180 / np.pi
            
            # Détection de mouvement significatif
            if np.mean(magnitudes) > 1.0:
                significant_movements += 1
                
                # Direction cohérente si l'écart-type des angles est faible
                if np.std(angles) < 40:  # 40 degrés de tolérance
                    coherent_movements += 1
                    flow_directions.append(np.mean(angles))
        
        # Mise à jour pour la prochaine itération
        old_frame = frame
        p0 = good_new.reshape(-1, 1, 2)
    
    if significant_movements == 0:
        return 0.0
    
    # Score basé sur la cohérence et l'importance du mouvement
    coherence_ratio = coherent_movements / max(1, significant_movements)
    
    # Direction cohérente entre frames?
    direction_consistency = 0
    if len(flow_directions) > 1:
        # Vérification si le mouvement suit une tendance claire (même direction)
        direction_consistency = 1.0 - min(1.0, np.std(flow_directions) / 90)
    
    return min(1.0, coherence_ratio * 0.7 + direction_consistency * 0.3)

# 1. Forme animale (utile pour détecter renard, chien)
@register_test(weight_range=(0, 10), pow=1.0)
def test_forme_animale(frames_data):
    # Pour assurer la compatibilité avec l'ancien code
    gray_frames = frames_data[0] if isinstance(frames_data, tuple) else frames_data
    if len(gray_frames) < 2:
        return 0.0
    total_score = 0
    scored_frames = 0
    prev_centroid = None
    for i in range(1, len(gray_frames)):
        delta = cv2.absdiff(gray_frames[i - 1], gray_frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:
                continue
            score = 0
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 1.5 < aspect_ratio < 4.0:
                score += 2
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * 3.1416 * area / (perimeter * perimeter)
                if 0.1 < circularity < 0.5:
                    score += 1
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if prev_centroid:
                    dx = cx - prev_centroid[0]
                    dy = cy - prev_centroid[1]
                    dist = (dx**2 + dy**2)**0.5
                    if 10 < dist < 80:
                        score += 1
                prev_centroid = (cx, cy)
            if score > best_score:
                best_score = score
        if best_score > 0:
            total_score += best_score
            scored_frames += 1
    if scored_frames == 0:
        return 0.0
    avg_score = total_score / scored_frames
    return min(avg_score / 5.0, 1.0)

# 2. Objets mobiles (détection brute)
@register_test(weight_range=(0, 7), pow=1.0)
def test_objets_mobiles(frames_data):
    # Pour assurer la compatibilité avec l'ancien code
    gray_frames = frames_data[0] if isinstance(frames_data, tuple) else frames_data
    if len(gray_frames) < 2:
        return 0.0
    objets_detectes = 0
    total = 0
    for i in range(1, len(gray_frames)):
        delta = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.contourArea(c) > 800 for c in contours):
            objets_detectes += 1
        total += 1
    if total == 0:
        return 0.0
    return min((objets_detectes / total) * 2, 1.0)

# 3. Dispersion dans la scène (traversée vs local)
@register_test(weight_range=(0, 10), pow=1.2)
def test_dispersion_mouvement(frames_data):
    # Pour assurer la compatibilité avec l'ancien code
    gray_frames = frames_data[0] if isinstance(frames_data, tuple) else frames_data
    if len(gray_frames) < 2:
        return 0.0
    h, w = gray_frames[0].shape
    centroïdes = []
    for i in range(1, len(gray_frames)):
        delta = cv2.absdiff(gray_frames[i - 1], gray_frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroïdes.append((cx, cy))
    if len(centroïdes) < 2:
        return 0.0
    xs = [pt[0] for pt in centroïdes]
    ys = [pt[1] for pt in centroïdes]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    scene_area = w * h
    occupation_ratio = bbox_area / scene_area
    return min(occupation_ratio * 5, 1.0)

@register_test(weight_range=(0, 10), pow=1.5)
def test_persistant_vs_intermittent(frames_data):
    # Pour assurer la compatibilité avec l'ancien code
    gray_frames = frames_data[0] if isinstance(frames_data, tuple) else frames_data
    if len(gray_frames) < 5:  # besoin d'un minimum de frames
        return 0.0
    sequence_count = 0
    current = 0
    for i in range(1, len(gray_frames)):
        delta = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        moving = cv2.countNonZero(thresh) > 1000
        if moving:
            current += 1
        else:
            if current >= 3:
                sequence_count += 1
            current = 0
    return min(sequence_count / 3.0, 1.0)

# 5. Ratio de trajectoire (distance parcourue dans la scène)
@register_test(weight_range=(0, 8), pow=1.2)
def test_longueur_trajectoire(frames_data):
    # Pour assurer la compatibilité avec l'ancien code
    gray_frames = frames_data[0] if isinstance(frames_data, tuple) else frames_data
    if len(gray_frames) < 2:
        return 0.0
    points = []
    for i in range(1, len(gray_frames)):
        delta = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    if len(points) < 2:
        return 0.0
    dist = sum(((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)**0.5 for i in range(1, len(points)))
    max_possible = ((gray_frames[0].shape[0]**2 + gray_frames[0].shape[1]**2)**0.5) * len(points)
    return min(dist / max_possible * 3, 1.0)

@apply_pow(pow=6)
def compute_motion_score(frames_data, video_name):
    score = 0
    print(f"\nVidéo analysée : {video_name}")
    for test_func in motion_tests:
        raw_score = test_func(frames)
        raw_score = min(max(raw_score, 0.0), 1.0)
        raw_score = raw_score ** test_func._pow
        min_w, max_w = test_func._weight_range
        if max_w == min_w:
            mapped = 0
        else:
            mapped = int(min_w + raw_score * (max_w - min_w))
        print(f"  {test_func.__name__:<35} [range {min_w}-{max_w}, pow={test_func._pow}] → {mapped} (brut={raw_score:.2f})")
        score += mapped
    print(f"  ➜ Score total final (avant pow) : {score}")
    return score

def score_video_file(path):
    """
    Analyse une vidéo et calcule un score d'intérêt basé sur plusieurs algorithmes
    de détection avancée
    """
    base_name = os.path.basename(path)
    gray_frames, color_frames = extract_frames(path)
    
    if not gray_frames:
        raise ValueError(f"Impossible de lire la vidéo : {path}")
        
    frames_data = (gray_frames, color_frames)
    score = compute_motion_score(frames_data, base_name)
    return score

def rescore_videos(directory="./captures"):
    avi_files = [f for f in os.listdir(directory) if f.lower().endswith(".avi")]
    avi_files = sorted(avi_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    for file in avi_files:
        original_path = os.path.join(directory, file)
        base_name = strip_score_prefix(file)
        clean_path = os.path.join(directory, base_name)
        if original_path != clean_path:
            os.rename(original_path, clean_path)
        try:
            score = score_video_file(clean_path)
        except ValueError as e:
            print(f"⚠️  {e}")
            continue
        new_name = f"{score}_{base_name}"
        new_path = os.path.join(directory, new_name)
        os.rename(clean_path, new_path)
        print(f"✅ {base_name} renommé en {new_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rescore .avi videos in a directory or a single video file.")
    parser.add_argument("input", nargs="?", default="./captures", help="Dossier ou fichier à analyser (défaut: ./captures)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        rescore_videos(args.input)
    elif os.path.isfile(args.input):
        try:
            score = score_video_file(args.input)
            print(f"Score obtenu pour {args.input} : {score}")
        except ValueError as e:
            print(f"⚠️  {e}")
    else:
        print(f"Entrée invalide : {args.input}")
