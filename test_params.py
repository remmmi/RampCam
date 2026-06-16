#!/usr/bin/env python3
"""
Script pour tester les paramètres de détection en temps réel
"""
import cv2
import numpy as np
import requests
import time
import argparse

def fetch_image(url, auth):
    try:
        response = requests.get(url, auth=auth, timeout=5)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return frame
    except requests.RequestException as e:
        print(f"Erreur récupération image : {e}")
        return None

def test_detection_params():
    parser = argparse.ArgumentParser(description="Test des paramètres de détection")
    parser.add_argument("--sensitivity", type=int, default=100, help="Seuil de détection (1-255)")
    parser.add_argument("--min-area", type=int, default=200, help="Surface minimale de détection")
    parser.add_argument("--duration", type=int, default=60, help="Durée du test en secondes")
    args = parser.parse_args()

    url = "http://192.168.1.111/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"
    auth = ('admin', 'password')
    
    prev_frame = None
    start_time = time.time()
    detection_count = 0
    
    print(f"Test avec sensibilité={args.sensitivity}, surface_min={args.min_area}")
    print(f"Durée du test: {args.duration}s")
    print("=" * 50)
    
    while time.time() - start_time < args.duration:
        frame = fetch_image(url, auth)
        if frame is None:
            time.sleep(0.2)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            time.sleep(0.2)
            continue

        delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(delta, args.sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = [cv2.contourArea(c) for c in contours]
        max_area = max(contour_areas) if contour_areas else 0
        mouvement = any(cv2.contourArea(c) >= args.min_area for c in contours)
        
        elapsed = int(time.time() - start_time)
        if mouvement:
            detection_count += 1
            print(f"[{elapsed:3d}s] MOUVEMENT DÉTECTÉ! Contours: {len(contours)}, Surface max: {max_area:.0f}")
        elif len(contours) > 0:
            print(f"[{elapsed:3d}s] Contours: {len(contours)}, Surface max: {max_area:.0f} (< {args.min_area})")
        
        prev_frame = gray
        time.sleep(0.2)
    
    print("=" * 50)
    print(f"Test terminé: {detection_count} détections en {args.duration}s")
    if detection_count == 0:
        print("CONSEIL: Réduisez --sensitivity ou --min-area pour plus de sensibilité")
    elif detection_count > 10:
        print("CONSEIL: Augmentez --sensitivity ou --min-area pour moins de fausses alertes")

if __name__ == '__main__':
    test_detection_params()
