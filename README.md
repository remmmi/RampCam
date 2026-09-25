# RampCam

Motion surveillance application for an IP camera watching a garage ramp.

RampCam polls the camera's JPEG stream in a loop, detects motion with an adaptive
background model (MOG2), confirms detections with YOLO, records a video (in-memory
pre-buffer + live footage), shows a tkinter video window, plays a horn sound and
sends an image notification through [ntfy](https://ntfy.sh).

## How it works

Detection runs in two stages to keep false positives low:

1. **Motion stage (MOG2)** - an adaptive background subtractor extracts moving
   regions from each frame. Corrupted frames (neon-colored camera glitches) are
   filtered out, and motion outside the surveillance mask is ignored.
2. **Confirmation stage (YOLO)** - when motion is found, a burst of frames is
   analyzed with yolov4-tiny. An alert fires only if a watched class (person,
   vehicle, animal) is detected inside the mask, overlapping a moving region,
   and not identified as a static object.

Additional safeguards:

- **Static object memory** - an object confirmed several times at the same spot
  (parked car, dropped item) is muted until it moves or is forgotten after a
  timeout. This prevents repeated alerts from reflections on a parked vehicle.
- **Startup warm-up** - detections are suppressed for a short period after
  startup while the background model stabilizes.
- **Recording post-processing** - videos that are too small or score too low on
  a motion metric are deleted automatically, and recordings older than a
  configurable age are cleaned up.
- **Smart notification frame** - the ntfy notification picks the best frame from
  several candidates using a hybrid motion + YOLO score. Notifications are
  rate-limited and silenced at night.

## Surveillance mask

`masque.png` defines the monitored zones: **green** pixels (0,255,0) are watched,
**red** pixels (255,0,0) are ignored. The mask is resized on the fly to match the
frame size. Motion outside the mask never triggers anything.

## Requirements

- Python 3 with a virtual environment providing `opencv-python`, `numpy`,
  `requests` and `Pillow`
- An X display (the video window uses tkinter; `xdotool` is used to restore
  window focus)
- yolov4-tiny model files in `models/` (`.cfg`, `.weights`, `coco.names`),
  loaded through `cv2.dnn` on CPU

## Configuration

Camera credentials and the ntfy topic live in `config_local.py`, which is not
tracked by git:

```bash
cp config_local.example.py config_local.py
# then edit CAMERA_URL, CAMERA_AUTH and NTFY_TOPIC
```

All other settings (MOG2 and YOLO thresholds, anti-false-positive tuning,
durations, paths) are constants at the top of `detect.py`.

## Usage

```bash
# Run detection (requires an X display)
python3 detect.py

# With sensitivity parameters
python3 detect.py --sensitivity 100 --min-area 500

# Test mode: simulates motion every 40s
python3 detect.py --test

# Restart cleanly (kills the running instance, then relaunches)
./restart_detect.sh
```

Useful CLI flags: `--fps`, `--prebuffer-secs`, `--live-secs`,
`--record-duration`, `--sensitivity` (diff threshold 1-255), `--min-area`
(minimum area in px2), `--max-age-hours` (video retention).

## Keeping it alive

Two redundant watchdogs restart `detect.py` if it stops:

- `cron_detect.sh` - meant to be called by cron every minute and at `@reboot`.
  Waits for X to be ready, avoids duplicates with `pgrep` and `flock`, and
  exports the display environment.
- `watch_detect.py` - an equivalent Python loop (pgrep + relaunch every 60s).

## Post-hoc classification

`classify.py` classifies and annotates captured videos with YOLO, either in a
windowed player or headless. It runs in dedicated virtual environments; see
`scripts/use-gui.sh` and `scripts/use-headless.sh` for setup instructions.

## Tests

The test suite is standalone (no framework needed); each file runs directly and
prints `N/N OK`:

```bash
python3 tests/test_detection_confirm.py
python3 tests/test_corrupt_frame.py
```

It covers the YOLO decision rule, motion/box overlap, static object memory,
warm-up behavior and the corrupted frame filter.

## Notes

- Code and log messages are in French.
- Data paths (videos, logs, annotated output) are set in `settings.conf`, section `[paths]`. Defaults live outside the project tree, in `~/Vidéos/RampCam_videos/`.
