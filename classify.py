#!/usr/bin/env python3
import os
import sys
import cv2
import time
import argparse
from collections import deque, Counter

# Paths relative to this script
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
COCO_NAMES = os.path.join(MODELS_DIR, "coco.names")
YOLO_CFG = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolov4-tiny.weights")


def load_class_names(names_path: str):
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Fichier classes introuvable: {names_path}")
    with open(names_path, "r", encoding="utf-8") as f:
        return [c.strip() for c in f.readlines() if c.strip()]


def build_net(cfg_path: str, weights_path: str):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"CFG introuvable: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Poids introuvables: {weights_path}")
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # CPU par défaut (stable). Si CUDA disponible, vous pouvez décommenter:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layer_names(net):
    layer_names = net.getLayerNames()
    # getUnconnectedOutLayers() renvoie indices 1-based
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


def has_highgui_support() -> bool:
    """Retourne True si OpenCV peut ouvrir une fenêtre (backend GUI dispo)."""
    try:
        test_name = "__cv_test_window__"
        cv2.namedWindow(test_name, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_name)
        return True
    except Exception:
        return False


def draw_predictions(frame, class_ids, confidences, boxes, class_names):
    for cls_id, conf, box in zip(class_ids, confidences, boxes):
        x, y, w, h = box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{class_names[cls_id]} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y), color, -1)
        cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / float(area_a + area_b - inter + 1e-6)


def process_video(video_path: str, net, out_names, class_names,
                  conf_thr: float, nms_thr: float, width: int, height: int, show_fps: bool,
                  display: bool, save: bool, out_dir: str, fourcc: str, out_fps: float,
                  stable_frames: int, min_draw_conf: float, match_iou: float) -> bool:
    """Traite une vidéo et affiche les détections.

    Retourne True si l'utilisateur a demandé de quitter (touche 'q'), False sinon.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir la vidéo: {video_path}")
        return False

    win_name = f"YOLOv4-tiny COCO - {os.path.basename(video_path)}"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    writer = None
    # Historique des détections récentes: deque de longueur stable_frames
    recent = deque(maxlen=max(1, stable_frames))

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Redimension optionnel pour accélérer
        if width > 0 and height > 0:
            frame = cv2.resize(frame, (width, height))

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_names)

        H, W = frame.shape[:2]
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = int(scores.argmax())
                conf = float(scores[cls_id])
                if conf >= conf_thr:
                    cx, cy, bw, bh = det[0:4]
                    x = int((cx - bw/2) * W)
                    y = int((cy - bh/2) * H)
                    w = int(bw * W)
                    h = int(bh * H)
                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(cls_id)

        # Appliquer NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, max(conf_thr, min_draw_conf), nms_thr)
        if len(idxs) > 0:
            idxs = idxs.flatten()
            boxes = [boxes[i] for i in idxs]
            confidences = [confidences[i] for i in idxs]
            class_ids = [class_ids[i] for i in idxs]
        else:
            boxes, confidences, class_ids = [], [], []

        # Filtrer par confiance minimale d'affichage
        filt = [(cid, conf, box) for cid, conf, box in zip(class_ids, confidences, boxes) if conf >= min_draw_conf]
        class_ids, confidences, boxes = [t[0] for t in filt], [t[1] for t in filt], [t[2] for t in filt]

        # Ajouter les détections courantes à l'historique
        recent.append([(cid, conf, box) for cid, conf, box in zip(class_ids, confidences, boxes)])

        # Voting temporel: dessiner uniquement si la détection est stable sur >= stable_frames
        stable_cls, stable_conf, stable_boxes = [], [], []
        if len(recent) >= stable_frames:
            # Pour chaque détection courante, chercher les correspondances dans l'historique
            for cid, conf, box in recent[-1]:
                matches = []
                for fr in list(recent)[:-1]:
                    # meilleur match par frame précédente
                    best = None
                    best_iou = 0.0
                    for pcid, pconf, pbox in fr:
                        iouv = box_iou(box, pbox)
                        if iouv >= match_iou and iouv > best_iou:
                            best = (pcid, pconf, pbox, iouv)
                            best_iou = iouv
                    if best is not None:
                        matches.append(best)
                # Vote si assez de frames support
                if len(matches) >= stable_frames - 1:  # courant + >=(stable_frames-1) antérieures
                    # Vote de classe
                    cls_votes = Counter([cid] + [m[0] for m in matches])
                    best_cls, _ = cls_votes.most_common(1)[0]
                    # Moyenne de confiance des matches de la classe gagnante
                    confs = [conf] + [m[1] for m in matches if m[0] == best_cls]
                    mean_conf = sum(confs) / max(1, len(confs))
                    if mean_conf >= min_draw_conf:
                        # Moyenne des boîtes pour lisser
                        bx = [box] + [m[2] for m in matches if m[0] == best_cls]
                        if bx:
                            mx = int(sum(b[0] for b in bx) / len(bx))
                            my = int(sum(b[1] for b in bx) / len(bx))
                            mw = int(sum(b[2] for b in bx) / len(bx))
                            mh = int(sum(b[3] for b in bx) / len(bx))
                            stable_cls.append(best_cls)
                            stable_conf.append(mean_conf)
                            stable_boxes.append([mx, my, mw, mh])

        # Dessiner les boîtes stables si disponibles, sinon (au tout début) rien
        if stable_boxes:
            draw_predictions(frame, stable_cls, stable_conf, stable_boxes, class_names)

        # Initialisation du writer au premier frame si nécessaire
        if save and writer is None:
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(out_dir, f"annotated_{base}.mp4" if fourcc.lower()=="mp4v" else f"annotated_{base}.avi")
            H, W = frame.shape[:2]
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
            # Respecte le framerate source, y compris 10 fps; fallback 25 si inconnu
            fps_use = out_fps if (out_fps and out_fps > 0) else (fps_src if fps_src > 0 else 25.0)
            codec = cv2.VideoWriter_fourcc(*fourcc)
            writer = cv2.VideoWriter(out_path, codec, fps_use, (W, H))
            if not writer.isOpened():
                print(f"[WARN] Impossible d'ouvrir la sortie vidéo: {out_path}")
                save = False
            else:
                print(f"[INFO] Enregistrement de la vidéo annotée: {out_path}")

        if save and writer is not None:
            writer.write(frame)

        if show_fps:
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        if display:
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                if writer is not None:
                    writer.release()
                cv2.destroyWindow(win_name)
                return True  # demande de quitter
            elif key == ord('n'):
                break  # passer à la vidéo suivante

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyWindow(win_name)
    return False


def list_video_files(input_path: str, recursive: bool = False):
    exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4", ".AVI", ".MKV", ".MOV"}
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        if recursive:
            for root, _, fnames in os.walk(input_path):
                for f in fnames:
                    if os.path.splitext(f)[1] in exts:
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(input_path):
                p = os.path.join(input_path, f)
                if os.path.isfile(p) and os.path.splitext(f)[1] in exts:
                    files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Classify and display video(s) with YOLOv4-tiny COCO")
    parser.add_argument("input", help="Chemin d'un fichier vidéo ou d'un dossier contenant des vidéos")
    parser.add_argument("--conf", type=float, default=0.3, help="Seuil de confiance (def 0.3)")
    parser.add_argument("--nms", type=float, default=0.4, help="Seuil NMS (def 0.4)")
    parser.add_argument("--width", type=int, default=640, help="Largeur de redimensionnement (0 pour garder)")
    parser.add_argument("--height", type=int, default=0, help="Hauteur de redimensionnement (0 pour garder)")
    parser.add_argument("--fps", action="store_true", help="Afficher le FPS")
    parser.add_argument("-r", "--recursive", action="store_true", help="Parcourir récursivement le dossier")
    parser.add_argument("--no-display", action="store_true", help="Désactiver l'affichage des fenêtres (headless)")
    parser.add_argument("--save", action="store_true", help="Sauvegarder la vidéo annotée")
    parser.add_argument("--out-dir", default=os.path.join(APP_DIR, "captures_annotated"), help="Dossier de sortie des vidéos annotées")
    parser.add_argument("--fourcc", default="mp4v", help="Codec de sortie (mp4v ou XVID)")
    parser.add_argument("--out-fps", type=float, default=0.0, help="FPS de sortie (0 = utiliser FPS source ou 25)")
    parser.add_argument("--stable-frames", type=int, default=5, help="Nombre de frames pour stabiliser une détection (vote)")
    parser.add_argument("--min-draw-conf", type=float, default=0.4, help="Confiance minimale pour afficher une box")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU minimum pour considérer qu'une box correspond entre frames")
    args = parser.parse_args()

    # Si width>0 et height==0, conserve ratio 16:9 approximatif
    if args.width > 0 and args.height == 0:
        args.height = int(args.width * 9 / 16)

    files = list_video_files(args.input, args.recursive)
    if not files:
        print("Aucune vidéo trouvée.")
        sys.exit(1)

    # Charger le modèle une fois
    class_names = load_class_names(COCO_NAMES)
    net = build_net(YOLO_CFG, YOLO_WEIGHTS)
    out_names = get_output_layer_names(net)

    # Détection GUI globale (une seule fois)
    gui_ok = False if args.no_display else has_highgui_support()
    if not gui_ok:
        # Message unique avec instructions
        print("[INFO] Environnement sans GUI: exécution en mode headless (pas d'affichage).")
        print("[INFO] Pour sauvegarder et visualiser les vidéos annotées:")
        print("       - Exécutez: python3 app/classify.py <dossier_ou_fichier> --no-display --save --out-dir app/captures_annotated")
        print("       - Les vidéos annotées seront dans: app/captures_annotated/")
        print("       - Lecture: ffplay app/captures_annotated/annotated_*.mp4")
        print("[INFO] Pour activer l'affichage fenêtré sur Ubuntu/Debian:")
        print("       sudo apt-get install -y libgtk-3-0 libgtk-3-dev python3-tk ffmpeg")
        print("       pip uninstall -y opencv-python-headless && pip install opencv-python")

    print(f"Lecture de {len(files)} vidéo(s). Touche 'n' pour passer, 'q' pour quitter.")
    for i, vp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {vp}")
        user_quit = process_video(vp, net, out_names, class_names,
                                  args.conf, args.nms, args.width, args.height, args.fps,
                                  display=gui_ok, save=args.save,
                                  out_dir=args.out_dir, fourcc=args.fourcc, out_fps=args.out_fps,
                                  stable_frames=max(1, args.stable_frames),
                                  min_draw_conf=args.min_draw_conf,
                                  match_iou=args.match_iou)
        if user_quit:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")
