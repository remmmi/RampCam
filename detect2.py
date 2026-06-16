#!/usr/bin/env python3
import cv2
import numpy as np
import requests
import time
import argparse
import os
import datetime
import subprocess
from threading import Thread, Lock, Event
from collections import deque
from queue import Queue
from rate_mov2 import score_video_file, initialize_detectors, create_bg_subtractor, test_detection_humaine, test_detection_multiclasse, detect_objects_yolo
import tkinter as tk
from PIL import Image, ImageTk

video_recording = False
video_lock = Lock()
# Stocke à la fois les frames en couleur et en niveaux de gris
prebuffer_frames = deque(maxlen=25)  # 5 sec @ 5 FPS
prebuffer_gray_frames = deque(maxlen=25)  # 5 sec @ 5 FPS
last_trigger_time = 0

ui_queue = Queue()
stop_event = Event()

def log(message):
    print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), message)

def fetch_image(url, auth):
    try:
        response = requests.get(url, auth=auth, timeout=5)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return frame
    except requests.RequestException as e:
        log("Erreur récupération image : " + str(e))
        return None

def beep(n, delay=8, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "klaxon.aac")):
    def _beep():
        for _ in range(n):
            subprocess.Popen([
                "ffplay", "-nodisp", "-autoexit", path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(delay)
    Thread(target=_beep, daemon=True).start()

def create_buffer_video_window(buffer_frames, duration=20):
    frames = list(buffer_frames)
    if not frames:
        return None
    
    # Détecter les objets dans la dernière image
    detectors = initialize_detectors()
    if 'yolo' in detectors:
        try:
            last_frame = frames[-1].copy()
            detections = detect_objects_yolo(last_frame, detectors, conf_threshold=0.4)
            
            # Ajouter des annotations pour toutes les détections
            for class_name, confidence, box in detections:
                x, y, w, h = box
                cv2.rectangle(last_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(last_frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Remplacer la dernière frame par celle avec annotations
            frames[-1] = last_frame
        except Exception as e:
            print(f"Erreur annotation: {e}")
        
    # Ajoutons un marqueur pour indiquer que c'est la détection avancée multi-classes
    for frame in frames:
        cv2.putText(frame, "Detection multi-classes", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   
    return {'frames': frames, 'duration': duration, 'idx': 0}

def record_video(url, auth, prebuffer, duration=30, fps=5, output_dir="./captures2"):
    global video_recording
    with video_lock:
        if video_recording:
            return
        video_recording = True

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    filename = os.path.join(output_dir, f"descente_{timestamp}.avi")

    if not prebuffer:
        log("Erreur : buffer de préenregistrement vide")
        with video_lock:
            video_recording = False
        return

    height, width, _ = prebuffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in prebuffer:
        out.write(frame)

    end_time = time.time() + duration
    while time.time() < end_time:
        frame = fetch_image(url, auth)
        if frame is not None:
            out.write(frame)
        time.sleep(1.0 / fps)

    out.release()
    log(f"Vidéo terminée, application des algorithmes avancés: {filename}")

    try:
        size_kb = os.path.getsize(filename) / 1024
        if size_kb < 420:
            os.remove(filename)
            log(f"Vidéo supprimée : {filename} (taille trop petite : {int(size_kb)} Ko)")
        else:
            # Utilise notre nouveau système d'évaluation amélioré
            score = score_video_file(filename)
            
            # Score réduit car YOLO est plus sensible et détecte mieux les objets
            if score < 120000:  # Seuil réduit pour tenir compte de la détection multi-classes
                os.remove(filename)
                log(f"Vidéo supprimée : {filename} (score insuffisant avec détection multi-classes : {score})")
            else:
                # Sauvegarde des annotations dans un fichier texte associé à la vidéo
                try:
                    # Analyser la vidéo pour extraire les classes détectées
                    cap = cv2.VideoCapture(filename)
                    step = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 10)  # Analyser 10 frames
                    if step < 1:
                        step = 1
                        
                    detectors = initialize_detectors()
                    detected_classes = set()
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        if frame_count % step == 0 and 'yolo' in detectors:
                            detections = detect_objects_yolo(frame, detectors, conf_threshold=0.35)
                            for class_name, _, _ in detections:
                                detected_classes.add(class_name)
                                
                        frame_count += 1
                        
                    cap.release()
                    
                    # Créer un fichier d'annotations
                    if detected_classes:
                        anno_filename = filename.replace(".avi", "_detections.txt")
                        with open(anno_filename, 'w') as f:
                            f.write(f"Détections dans {os.path.basename(filename)}\n")
                            f.write("-----------------------------\n")
                            f.write("\n".join(sorted(detected_classes)))
                        log(f"Objets détectés: {', '.join(sorted(detected_classes))}")
                except Exception as e:
                    log(f"Erreur génération annotations: {e}")
                
                log(f"Vidéo enregistrée : {filename} (score avec détection multi-classes = {score})")
    except Exception as e:
        log(f"Erreur post-enregistrement : {e}")

    with video_lock:
        video_recording = False

def clean_old_videos(folder="./captures2", max_age_hours=48, interval_seconds=600):
    while True:
        now = datetime.datetime.now()
        for file in os.listdir(folder):
            if file.startswith("descente_") and file.endswith(".avi"):
                try:
                    datetime_str = file[len("descente_"):-len(".avi")]
                    dt = datetime.datetime.strptime(datetime_str, "%Y_%m_%d-%H-%M-%S")
                    age = (now - dt).total_seconds() / 3600
                    if age > max_age_hours:
                        path = os.path.join(folder, file)
                        os.remove(path)
                        log(f"Supprimé (trop vieux) : {path}")
                except Exception as e:
                    log(f"Erreur analyse fichier {file} : {e}")
        time.sleep(interval_seconds)

def detection_thread(args):
    global last_trigger_time
    
    # Initialiser les détecteurs avancés d'OpenCV
    # Les détecteurs incluent maintenant YOLO pour les animaux et véhicules
    detectors = initialize_detectors()
    bg_subtractor = create_bg_subtractor()
    
    # Vérifier si YOLO est disponible
    has_yolo = 'yolo' in detectors
    if has_yolo:
        log("Détecteur YOLO chargé - Détection multi-classes active")
    else:
        log("Détecteur YOLO non disponible - Détection multi-classes désactivée")
    url = "http://192.168.1.111/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"  # URL de la caméra IP
    auth = ('admin', 'password')
    # Intervalle minimum entre 2 détections (évite déclenchements multiples)
    min_trigger_interval = 30  # secondes
    delay = 1.0 / args.fps

    log(f"Démarrage détection: seuil={args.sensitivity}, fps={args.fps}, min_area={args.min_area}, algorithmes=avancés")
    prev_frame = None
    detection_frames_buffer = []  # Buffer pour analyse avancée avec plusieurs frames
    os.makedirs("./captures2", exist_ok=True)
    Thread(target=clean_old_videos, args=("./captures2", args.max_age_hours), daemon=True).start()
    log("Surveillance démarrée...")

    while not stop_event.is_set():
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(delay)
            continue
            
        # Garder les frames en couleur pour l'analyse avancée
        prebuffer_frames.append(frame.copy())
        
        # Convertir et garder aussi en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prebuffer_gray_frames.append(gray)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue
            
        # Ajouter à notre buffer d'analyse
        detection_frames_buffer.append(gray)
        if len(detection_frames_buffer) > 10:  # Gardons 10 frames pour l'analyse
            detection_frames_buffer.pop(0)

        # 1. Détection traditionnelle par différence entre images (rapide, pour premier filtre)
        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, args.sensitivity, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        basic_motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < args.min_area:
                continue
            # Mouvement basique détecté
            basic_motion_detected = True
            break
            
        # 2. Si un mouvement basique est détecté, utiliser l'analyse plus avancée
        if basic_motion_detected and len(detection_frames_buffer) >= 5:
            # Utilisons la détection humaine avancée si nous avons assez de frames
            color_frames_sample = list(prebuffer_frames)[-5:]
            gray_frames_sample = list(prebuffer_gray_frames)[-5:]
            
            # Si le background subtractor détecte un mouvement significatif
            fg_mask = bg_subtractor.apply(gray)
            movement_ratio = cv2.countNonZero(fg_mask) / (fg_mask.shape[0] * fg_mask.shape[1])
            
            if movement_ratio > 0.005:  # Si au moins 0.5% de l'image montre du mouvement
                frames_data = (gray_frames_sample, color_frames_sample)
                
                # Détection humaine (HOG + Cascade)
                human_score = test_detection_humaine(frames_data)
                
                # Détection multi-classe (YOLO pour animaux, véhicules, etc.)
                multiclass_score = test_detection_multiclasse(frames_data)
                
                # On déclenche si humain OU animaux/véhicules détectés
                if human_score > 0.15 or multiclass_score > 0.2:
                    motion_detected = True
                    
                    if human_score > 0.15:
                        log(f"Détection avancée: présence humaine probable (score={human_score:.2f})")
                    
                    if multiclass_score > 0.2:
                        # Vérifier les détections spécifiques dans la dernière frame en couleur
                        if 'yolo' in detectors:
                            last_frame = color_frames_sample[-1]
                            detections = detect_objects_yolo(last_frame, detectors, conf_threshold=0.4)
                            if detections:
                                classes_detected = ', '.join([d[0] for d in detections])
                                log(f"Détection multi-classe: {classes_detected} (score global={multiclass_score:.2f})")

        current_time = time.time()
        if motion_detected and (current_time - last_trigger_time) > min_trigger_interval:
            log("⚠️ Mouvement détecté! Détection anim./véhic. (ALERTES DÉSACTIVÉES)")
            last_trigger_time = current_time
            # Alertes désactivées - pas de beep ni d'enregistrement vidéo
            # On garde juste la détection pour le monitoring

        prev_frame = gray
        time.sleep(delay)

def parse_args():
    parser = argparse.ArgumentParser(description="Détection de mouvement camera avec détection multi-classe")
    parser.add_argument("--sensitivity", type=int, default=30,
                        help="Sensibilité de détection (20-60)")
    parser.add_argument("--min-area", type=int, dest="min_area", default=500,
                        help="Taille minimale zone de mouvement")
    parser.add_argument("--fps", type=int, default=5,
                        help="Images par seconde")
    parser.add_argument("--no-gui", action="store_true", dest="no_gui",
                        help="Désactiver l'interface graphique")
    parser.add_argument("--max-age", type=int, default=48, dest="max_age_hours",
                        help="Durée max de conservation des vidéos (heures)")
    parser.add_argument("--disable-yolo", action="store_true", dest="disable_yolo",
                        help="Désactiver la détection multi-classe YOLO")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Créer le répertoire de captures s'il n'existe pas déjà
    os.makedirs("./captures2", exist_ok=True)

    gui_enabled = False
    if os.environ.get("DISPLAY"):
        try:
            test_root = tk.Tk()
            test_root.withdraw()
            test_root.destroy()
            gui_enabled = True
        except tk.TclError:
            log("Session graphique détectée mais inutilisable (TclError)")
    else:
        log("Aucune session graphique active (DISPLAY non défini)")

    # URL et authentification de la caméra
    url = "http://192.168.1.111/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"
    auth = ('admin', 'password')
    
    detection_thread_instance = Thread(target=detection_thread, args=(args,), daemon=True)
    detection_thread_instance.start()

    if gui_enabled:
        root = tk.Tk()
        root.title("Surveillance Caméra - Visualisation Permanente")
        
        # Créer une fenêtre principale au lieu de la cacher
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panneau pour l'affichage vidéo
        video_panel = tk.Label(main_frame)
        video_panel.pack(fill=tk.BOTH, expand=True)
        
        # Panneau d'information
        info_panel = tk.Label(main_frame, text="Mode visualisation permanente - Alertes désactivées", 
                             bg="yellow", fg="black")
        info_panel.pack(fill=tk.X)
        
        # Fonction pour mettre à jour l'image en continu
        def update_live_view():
            if stop_event.is_set():
                return
                
            # Récupérer la dernière image du buffer si disponible
            if prebuffer_frames and len(prebuffer_frames) > 0:
                frame = prebuffer_frames[-1].copy()
                
                # Ajouter des annotations si des détections sont en cours
                detectors = initialize_detectors()
                if 'yolo' in detectors:
                    try:
                        detections = detect_objects_yolo(frame, detectors, conf_threshold=0.4)
                        
                        # Ajouter des annotations pour toutes les détections
                        for class_name, confidence, box in detections:
                            x, y, w, h = box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Erreur annotation: {e}")
                
                # Convertir l'image pour l'affichage Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                video_panel.config(image=photo)
                video_panel.image = photo  # Garder une référence
            
            # Planifier la prochaine mise à jour
            root.after(200, update_live_view)  # Rafraîchir à 5 FPS
        
        # Démarrer la mise à jour de l'affichage
        update_live_view()
        
        # Fonction pour vérifier les messages dans la queue (pour compatibilité)
        def check_queue():
            try:
                while True:
                    ui_queue.get_nowait()
                    ui_queue.task_done()
            except Exception:
                pass
            root.after(100, check_queue)
        
        check_queue()

        def on_closing():
            stop_event.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    else:
        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nProgramme arrêté par l'utilisateur")
