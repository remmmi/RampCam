#!/usr/bin/env python3
"""Extrait une image par seconde de chaque video dans captures/."""

import os
import cv2

CAPTURES_DIR = "/home/m/Bureau/camera-venv/app/captures"
TMP_DIR = "/home/m/Bureau/camera-venv/app/tmp"
MAX_SECONDS = 15

def extract_frames():
    # Creer les dossiers tmp/1 a tmp/15
    for sec in range(1, MAX_SECONDS + 1):
        os.makedirs(os.path.join(TMP_DIR, str(sec)), exist_ok=True)

    # Lister les videos (pas les sous-dossiers)
    videos = [f for f in os.listdir(CAPTURES_DIR)
              if f.endswith('.avi') and os.path.isfile(os.path.join(CAPTURES_DIR, f))]

    print(f"Trouvé {len(videos)} videos")

    for video_name in sorted(videos):
        video_path = os.path.join(CAPTURES_DIR, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Erreur ouverture: {video_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10  # fallback

        base_name = os.path.splitext(video_name)[0]

        for sec in range(1, MAX_SECONDS + 1):
            frame_number = int(sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                output_path = os.path.join(TMP_DIR, str(sec), f"{sec}_{base_name}.jpg")
                cv2.imwrite(output_path, frame)
            else:
                break  # fin de la video

        cap.release()
        print(f"Traité: {video_name}")

    print("Terminé")

if __name__ == "__main__":
    extract_frames()
