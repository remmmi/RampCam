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
from rate_mov import score_video_file
import tkinter as tk
from PIL import Image, ImageTk

video_recording = False
video_lock = Lock()
prebuffer_frames = deque(maxlen=25)  # 5 sec @ 5 FPS
last_trigger_time = 0

ui_queue = Queue()
stop_event = Event()

def log(message):
    now = datetime.datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    log_line = f"{timestamp} {message}"
    print(log_line)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = now.strftime("%Y.%m.%d.detect.log")
    log_path = os.path.join(log_dir, log_filename)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[LOGGING ERROR] Impossible d’écrire dans le fichier log: {e}")

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
    return {'frames': frames, 'duration': duration, 'idx': 0}

def record_video(url, auth, prebuffer, duration=30, fps=5, output_dir="./app/captures"):
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

    try:
        size_kb = os.path.getsize(filename) / 1024
        if size_kb < 420:
            os.remove(filename)
            log(f"Vidéo supprimée : {filename} (taille trop petite : {int(size_kb)} Ko)")
        else:
            score = score_video_file(filename)
            if score < 200000:
                os.remove(filename)
                log(f"Vidéo supprimée : {filename} (score insuffisant : {score})")
            else:
                log(f"Vidéo enregistrée : {filename} (score = {score})")
    except Exception as e:
        log(f"Erreur post-enregistrement : {e}")

    with video_lock:
        video_recording = False

def clean_old_videos(folder="./captures", max_age_hours=48, interval_seconds=600):
    while not stop_event.is_set():
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
    url = "http://192.168.1.111/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"
    auth = ('admin', 'password')
    delay = 1.0 / args.fps
    prev_frame = None

    os.makedirs("./captures", exist_ok=True)
    Thread(target=clean_old_videos, args=("./captures", args.max_age_hours), daemon=True).start()
    log("Surveillance démarrée...")

    while not stop_event.is_set():
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(delay)
            continue

        prebuffer_frames.append(frame.copy())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            time.sleep(delay)
            continue

        delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(delta, args.sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug: calculer les surfaces des contours
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_area = max(contour_areas) if contour_areas else 0
        mouvement = any(cv2.contourArea(c) >= args.min_area for c in contours)
        
        # Debug périodique (toutes les 60 secondes)
        current_time = time.time()
        if int(current_time) % 60 == 0 and int(current_time) != getattr(detection_thread, 'last_debug_time', 0):
            detection_thread.last_debug_time = int(current_time)
            log(f"[DEBUG] Contours: {len(contours)}, Surface max: {max_area}, Seuil: {args.min_area}, Sensibilité: {args.sensitivity}")
        
        if mouvement and (current_time - last_trigger_time) > 30:
            log("Mouvement détecté")
            last_trigger_time = current_time
            beep(1)
            Thread(target=record_video, args=(url, auth, list(prebuffer_frames)), daemon=True).start()
            ui_queue.put(('show_video', create_buffer_video_window(prebuffer_frames, duration=40)))
        elif mouvement:
            log(f"[DEBUG] Mouvement détecté mais ignoré (délai de {int(current_time - last_trigger_time)}s < 30s)")
        elif contours:
            log(f"[DEBUG] Contours détectés mais trop petits - Surface max: {max_area} < {args.min_area}")

        prev_frame = gray
        time.sleep(delay)

def show_buffer_video(parent, video_data):
    frames = video_data['frames']
    duration = video_data['duration']

    video_window = tk.Toplevel(parent)
    video_window.title("Alerte vidéo")
    video_window.attributes("-topmost", True)

    h, w, _ = frames[0].shape
    video_window.geometry(f"{w}x{h}+100+100")

    panel = tk.Label(video_window)
    panel.pack()

    def close(event=None):
        if video_window.winfo_exists():
            video_window.destroy()

    video_window.bind("<Button-1>", close)

    def update_frame():
        if not video_window.winfo_exists():
            return
        idx = video_data['idx']
        if idx >= len(frames):
            video_window.destroy()
            return

        frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        panel.config(image=photo)
        panel.image = photo

        video_data['idx'] += 1
        video_window.after(int(1000 / 5), update_frame)

    video_window.after(0, update_frame)
    video_window.after(duration * 1000, close)

def main():
    parser = argparse.ArgumentParser(description="Détection de mouvement caméra IP avec interface Tkinter")
    parser.add_argument("--fps", type=int, default=5, help="Fréquence d'analyse des images (défaut: 5)")
    parser.add_argument("--sensitivity", type=int, default=120, help="Seuil de détection (1-255, défaut: 180)")
    parser.add_argument("--min-area", type=int, default=1500, help="Surface minimale de détection en pixels")
    parser.add_argument("--max-age-hours", type=int, default=72, help="Âge max des vidéos avant suppression")
    args = parser.parse_args()

    root = tk.Tk()
    root.withdraw()

    def check_queue():
        try:
            while True:
                if not root.winfo_exists():
                    return
                action, data = ui_queue.get_nowait()
                if action == 'show_video' and data:
                    show_buffer_video(root, data)
                ui_queue.task_done()
        except:
            pass
        if root.winfo_exists():
            root.after(100, check_queue)

    detection_thread_instance = Thread(target=detection_thread, args=(args,), daemon=True)
    detection_thread_instance.start()
    check_queue()

    def on_closing():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nProgramme arrêté par l'utilisateur")
