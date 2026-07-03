# Sentinel — Project Journal

> Living memory of the project. Re-read (or paste to an assistant) at the start
> of a session to restore full context. Update after each significant step.

Last updated: 2026-07-03

---

## 0. Pitch & goal

Autonomous surveillance mini-tower inspired by Anduril's Sentry Towers.
**Detection only, no weapons.** Workflow: an mmWave radar detects motion and
gives azimuth/range (slew-to-cue) → the camera observes that zone → an AI model
analyses the live video stream and classifies what it sees (human / vehicle /
animal). Real-time tactical dashboard.

Personal goal: portfolio piece to target defense/aerospace engineering
(Anduril, Thales, MBDA, Dassault, SpaceX). Built by Léo, MPSI student, strong in
software, learning sensors/embedded/ML.

---

## 1. Project status (phases)

- **Phase 1 — Fixed radar detector + real-time dashboard**: ✅ WORKING
  Radar physically validated, async UART driver, WebSocket dashboard, SQLite
  persistence, deployed headless on the Pi.
- **Phase 2 — Camera + AI + auto-pointing**: ⏳ IN PROGRESS
  Camera wired and capturing. Slew-to-cue V1 done (radar → auto photo).
  Vision foundation built (model comparator + real-time detection viewer).
  Model chosen: **YOLO11n**. Next: run YOLO on the live Pi Camera stream.
- **Phase 3 — Long range + fusion + multi-target tracking**: 🔜 planned.

---

## 2. Hardware (what is actually owned)

- Raspberry Pi 5 (8 GB) + Active Cooler + official 27W USB-C PSU.
- Hi-Link **HLK-LD2450** 24 GHz mmWave radar (up to 3 targets, distance + angle,
  ~8 m range — proximity layer).
- **HKJL** breakout board + JST cable, **HLK-CH340E** USB-UART converter.
- **Pi Camera Module 3** (Sony IMX708) via CSI (port J3 / CAM/DISP 0).
- SanDisk 64GB microSD, breadboard + jumpers, 2× CSI 15→22 pin cables.

**NOT owned**: no Hailo-8 / AI HAT (checked — it was only an Amazon page being
read aloud). No thermal/IR sensor. No pan/tilt servos yet. No extra radars yet.
→ YOLO runs on the **Pi 5 CPU only** (~2-5 FPS), which is enough for this use
case. Code is designed to accept a Hailo NPU later without a rewrite.

### Radar wiring (INVERTED Hi-Link colour convention — physically verified)
Black = 5V, Yellow = GND, White = radar TX → CH340E RXT, Red = radar RX →
CH340E TXD. TX/RX crossed. Never unplug the Pi without `sudo shutdown -h now`
(SD-card corruption risk). Pi 5: single STAT LED — green = running, solid red =
halted (safe to unplug).

### Ports
- Mac: radar at `/dev/cu.usbserial-11240`.
- Pi: radar at `/dev/ttyUSB0` (needs `usermod -aG dialout leo` once).

---

## 3. Software architecture

Repo: github.com/phirou/project-sentinel (public, MIT).
Mac: `~/Developer/project-sentinel` — Pi: `~/project-sentinel`.

```
sentinel/
  core/        models, event_bus, persistence, queries
  drivers/     ld2450 parser/simulator/uart driver, camera.py
  consumers/   camera_trigger.py (slew-to-cue)
  api/         FastAPI app + WebSocket
  main.py      asyncio orchestration
web/           tactical dashboard (HTML/CSS/JS)
ml/            from-scratch PyTorch pipeline (dataset, model, train) + compare_models.py
tools/
  detection_viewer/   FastAPI + MJPEG live YOLO viewer (model & source selectors)
  benchmark/          model benchmark script (FPS, detections, confidence)
docs/          architecture.md, JOURNAL.md
```

### Architecture principles
- **Async EventBus (pub/sub)**: drivers publish, consumers subscribe (console,
  SQLite, camera trigger, WebSocket). Adding a consumer never touches the pipeline.
- **Dependency inversion** on the radar source: simulator and real driver expose
  the same `stream()` interface.
- **asyncio** single loop; blocking camera capture offloaded via
  `asyncio.to_thread` so the loop never stalls.

### Notable bugs solved
- **Radar Y-axis convention**: LD2450 returns Y < 0 in front of the sensor
  (opposite of standard convention). Without the fix all targets showed at ~180°.
  Fixed with `atan2(x, -y)`, validated by physical measurement.
- **Asyncio shutdown**: Uvicorn intercepted SIGINT → zombie tasks. Fixed by
  disabling Uvicorn's signal handlers and managing shutdown explicitly.

### Slew-to-cue
`CameraTrigger` consumer: on radar detection → non-blocking camera capture,
throttled (5s cooldown), filename encodes target position (`T0_d2.3m_a-15deg`).

---

## 4. Vision / AI decisions

- **Compute**: dev on Mac M-series (MPS). Deploy inference on **Pi 5 CPU** (no NPU).
- **Pivot** (July): from single-photo classification to **continuous video-stream
  detection** with a pre-trained YOLO (detects 80 COCO classes: person, car,
  dog…), covering the "not just humans" need without training.
- **Model chosen: YOLO11n.** Benchmarked yolov8n / yolov8s / yolo11n over 5 test
  videos × 2 resolutions. yolo11n = fastest + cleanest detections. yolov8s
  eliminated (≈2× slower, more false positives — "hallucinated" horses/elephants).
- **Resolution trade-off**: 640px = fast but misses far targets (hard in forest);
  1280px = slower but sees distant people. Plan: 640px continuous, 1280px on a
  radar cue (leverages slew-to-cue instead of fighting it).
- **Custom CNN** (from-scratch PyTorch, ~155k params) kept for learning + a
  "built it, then used YOLO understanding what it does" portfolio story. Written,
  never trained (dataset still empty).
- **Dataset plan** (with Léo's father, data expert): FiftyOne + Open Images /
  COCO-2017 (`person`) + Places (scenes), ~6000 images, 50/50, 80/20 split,
  CC BY 4.0. Curation required (remove humans that appear in "non_human" scenes).

### Test videos (tools/detection_viewer/videos_test/, gitignored)
crowd_street_dense · street_cars_pedestrians_far · forest_autumn_2people_path ·
forest_2people_2dogs_tracking · forest_2people_hiking. Cover density, distance,
vegetation background (hard case), animals + tracking.

### Tools built
- `ml/compare_models.py`: offline model benchmark on a single video.
- `tools/detection_viewer/`: FastAPI + MJPEG viewer — live annotated feed, hot-
  swap model & video source, detection thumbnail gallery, live FPS, camera-input
  stub ready for the Pi Camera / RTSP.
- `tools/benchmark/`: full model×video×resolution benchmark, results.md output.

---

## 5. Working method (Léo's preferences)

- Give code in **blocks** with explanations; Léo **types it himself** for ML /
  vision (skill to acquire) — no blind copy-paste, validate each step.
- **Claude Code** is fine for plumbing (UI, analysis scripts), NOT for what Léo
  must learn.
- Always run/test before moving on.
- When a terminal is pasted and all is fine: short reply, no wall of text.
- Always mark 🍓 Pi vs 💻 Mac for each command.
- Don't push for breaks; if Léo wants to continue, continue.
- French comments in code. Modular, clean architecture (shown to recruiters).
- Prompts: Pi = `leo@sentinel`, Mac = `leos-MacBook-Air`.

For model choice on Claude Code prompts: Léo will pick Fable 5 vs Opus; the
assistant only flags in two words when a task genuinely needs the stronger model.

---

## 6. Deployment & ops

- Run radar (Pi): `ssh leo@sentinel.local` → `cd ~/project-sentinel` →
  `source .venv/bin/activate` → `python -m sentinel.main` →
  dashboard `http://sentinel.local:8000`.
- Run detection viewer (Mac): `uvicorn tools.detection_viewer.app:app --port 8100`
  → `http://localhost:8100`.
- Pi venv created with `--system-site-packages` (for `picamera2`).
- Lille Wi-Fi added to the Pi (usable away from home).
- systemd auto-start deliberately NOT enabled (manual control during dev).
- Git push from Pi needs a PAT; from Mac uses the SSH key (id_ed25519, passphrase).
- Videos and model weights are gitignored (keep the repo light — no large binaries).

---

## 7. Next steps

1. Wire **YOLO11n onto the live Pi Camera stream** (on the Pi, CPU). The viewer
   already has a camera-input stub — plug in the real camera. (Needs the Pi.)
2. Add **ByteTrack** tracking (stable IDs → count unique people/vehicles).
3. Build the dataset with Léo's father → first CNN training run.
4. Later modules on the vision core: intrusion zones, flow counting, **vehicle
   speed estimation** (detection + tracking + distance calibration — possible
   "sell to cities" angle, Phase 3).
5. Hardware later (when budget): pan/tilt servos, 2nd LD2450 for wider coverage,
   thermal/IR for night/forest, optional Hailo NPU.
6. Housekeeping: update the saved system prompt (remove Hailo/YOLOv8, add video
   pivot + YOLO11n); mobile-responsive dashboard; docs/ld2450-protocol.md.
