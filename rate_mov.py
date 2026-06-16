import cv2
import os
import re
from statistics import stdev

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
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        return []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames

# 1. Forme animale (utile pour détecter renard, chien)
@register_test(weight_range=(0, 10), pow=1.0)
def test_forme_animale(frames):
    if len(frames) < 2:
        return 0.0
    total_score = 0
    scored_frames = 0
    prev_centroid = None
    for i in range(1, len(frames)):
        delta = cv2.absdiff(frames[i - 1], frames[i])
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
def test_objets_mobiles(frames):
    if len(frames) < 2:
        return 0.0
    objets_detectes = 0
    total = 0
    for i in range(1, len(frames)):
        delta = cv2.absdiff(frames[i-1], frames[i])
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
def test_dispersion_mouvement(frames):
    if len(frames) < 2:
        return 0.0
    h, w = frames[0].shape
    centroïdes = []
    for i in range(1, len(frames)):
        delta = cv2.absdiff(frames[i - 1], frames[i])
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

# 4. Présence continue (mouvement sur frames consécutives)
@register_test(weight_range=(0, 10), pow=1.5)
def test_persistant_vs_intermittent(frames):
    if len(frames) < 5:
        return 0.0
    sequence_count = 0
    current = 0
    for i in range(1, len(frames)):
        delta = cv2.absdiff(frames[i-1], frames[i])
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
def test_longueur_trajectoire(frames):
    if len(frames) < 2:
        return 0.0
    points = []
    for i in range(1, len(frames)):
        delta = cv2.absdiff(frames[i-1], frames[i])
        thresh = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)[1]
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    if len(points) < 2:
        return 0.0
    dist = sum(((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)**0.5 for i in range(1, len(points)))
    max_possible = ((frames[0].shape[0]**2 + frames[0].shape[1]**2)**0.5) * len(points)
    return min(dist / max_possible * 3, 1.0)

@apply_pow(pow=6)
def compute_motion_score(frames, video_name):
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
    base_name = os.path.basename(path)
    frames = extract_frames(path)
    if not frames:
        raise ValueError(f"Impossible de lire la vidéo : {path}")
    score = compute_motion_score(frames, base_name)
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
