# Project Sentinel

> **Autonomous detection-only surveillance system** inspired by [Anduril Sentry Towers](https://www.anduril.com/) — combining mmWave radar, computer vision, and embedded AI on a Raspberry Pi 5.

[![Status](https://img.shields.io/badge/status-Phase%202%20in%20development-orange)](#development-phases)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Vision

Sentinel is a low-cost, fully autonomous **slew-to-cue** surveillance system: a 24 GHz mmWave radar continuously detects motion in its field of view, automatically cues a pan/tilt camera toward the target, captures an image, and runs on-device AI classification — all in under one second, all locally on a Raspberry Pi 5.

**Strictly detection-only. No weapons. No autonomous engagement.**

This project explores the architecture of modern defense surveillance systems (sensor fusion, real-time tactical display, edge AI) using consumer-grade hardware, as a learning vehicle for embedded systems and defense-tech engineering.

---

## System Architecture

```
                                    [ Web Dashboard ]
                                            ▲
                                            │ WebSocket (real-time)
                                            │
   ┌────────────┐    bytes     ┌────────────┴────────────┐    detection event
   │  LD2450    │──────────────▶│      EventBus           │────────────────────┐
   │  (radar)   │   UART 256k  │   (async pub/sub)       │                    │
   └────────────┘              └────────────┬────────────┘                    │
                                            │                                  ▼
                                            ▼                       ┌──────────────────┐
                                  ┌─────────────────┐                │  CameraTrigger   │
                                  │  SQLite logger  │                │  (slew-to-cue,   │
                                  │  (aiosqlite)    │                │   throttled)     │
                                  └─────────────────┘                └────────┬─────────┘
                                                                              │
                                                                              ▼
                                                                    ┌──────────────────┐
                                                                    │  Pi Camera 3     │
                                                                    │  (IMX708)        │
                                                                    └────────┬─────────┘
                                                                              │ image
                                                                              ▼
                                                                    ┌──────────────────┐
                                                                    │  CNN classifier  │
                                                                    │  (in progress)   │
                                                                    └──────────────────┘
```

The system is built around an **async pub/sub EventBus**: hardware drivers publish events (radar frames, parse errors), consumers subscribe to them (console logger, SQLite persistence, camera trigger, WebSocket broadcaster). Adding a new behaviour means adding a new consumer — never modifying the pipeline. This is the architectural pattern behind real defense sensor-fusion systems.

---

## Hardware

| Component | Role | Notes |
|---|---|---|
| Raspberry Pi 5 (8 GB) | Main compute | With Active Cooler, headless via SSH |
| Hi-Link HLK-LD2450 | 24 GHz mmWave radar | Tracks up to 3 targets, ±60° FoV, ~8 m range |
| HLK-CH340E | USB-UART converter | Bridges LD2450 to Pi USB at 256000 baud |
| Pi Camera Module 3 | Vision | Sony IMX708 sensor, CSI connection |
| (planned) Hailo-8 AI HAT | NPU for inference | Future ML inference acceleration |
| (planned) 2× servos | Pan/tilt camera mount | Slew-to-cue physical actuation |

---

## Software Stack

- **Python 3.12** with `asyncio` for all real-time pipeline orchestration
- **FastAPI + WebSocket** for the tactical dashboard backend
- **aiosqlite** for non-blocking detection persistence
- **picamera2** for camera control (lazy-imported for cross-machine development)
- **pyserial-asyncio** for the UART radar driver
- **PyTorch** (with Apple Silicon MPS backend on dev machine) for ML training
- **Pydantic v2** for typed data models with auto-serialization
- Vanilla **HTML/CSS/JS** for the dashboard (no framework — Lattice-style green-phosphor radar polar plot)
- **YAML** configuration with environment-specific overrides

---

## Development Phases

### Phase 1 — Static radar + real-time dashboard ✅

- Reverse-engineered Hi-Link LD2450 binary frame protocol (signed-magnitude convention)
- Asyncio UART driver with header sync and auto-reconnection
- EventBus-based pipeline (publisher/subscriber decoupling)
- SQLite persistence with WAL mode
- FastAPI + WebSocket backend
- Web dashboard with polar radar display, target list, telemetry
- Hardware-in-the-loop testing via interchangeable simulator/UART sources
- Full deployment on Raspberry Pi 5 (headless, autonomous)

### Phase 2 — Camera + AI classification ⏳ (current)

- ✅ Pi Camera Module 3 integration via `picamera2`
- ✅ Slew-to-cue: radar detection triggers a non-blocking camera capture
- ✅ Capture throttling (configurable cooldown) and filename encoding of radar position
- ⏳ CNN classifier (built from scratch in PyTorch) — architecture complete, dataset in progress
- ⏳ Transfer learning baseline (MobileNetV3) for comparison
- 🔜 Hailo NPU compilation pipeline for on-device inference
- 🔜 Pan/tilt motorized camera mount

### Phase 3 — Long range + sensor fusion 🔜

- Long-range radar (>500 m) integration
- Multi-radar 360° coverage (staring array architecture)
- Multi-target tracking with stable ID assignment
- Advanced tactical interface

---

## Quick Start

### On a development machine (Mac / Linux)

```bash
git clone https://github.com/phirou/project-sentinel.git
cd project-sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m sentinel.main
```

The dashboard is then available at <http://localhost:8000>. By default the system runs with a built-in simulator that emits synthetic radar frames at 10 Hz — no hardware required to explore the project.

### With real hardware (Raspberry Pi)

```bash
# On the Pi, after installation and venv setup with --system-site-packages
# (needed for picamera2 access):
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add user to dialout group (once) for serial port access
sudo usermod -aG dialout $USER
# Then log out and back in.

# Edit config/local.yaml to set radar.source: uart and uart_port: /dev/ttyUSB0
# Then launch:
python -m sentinel.main
```

The dashboard is accessible from any device on the same network at `http://sentinel.local:8000`.

---

## Notable Engineering Decisions

A few moments worth surfacing — both as documentation and as engineering judgment.

**Hi-Link Y-axis convention.** The LD2450 uses `Y < 0` for targets in front of the radar, the opposite of the standard mathematical convention. Without this correction, all detections appeared at ~180° (behind the radar). Diagnosed by physical reference measurements (target straight ahead, target left, target right), then fixed by inverting Y in `atan2(x, -y)`. Frame convention mismatches are one of the most common bug classes in robotics and remote sensing — this was a textbook case.

**Async signal propagation.** Uvicorn intercepted SIGINT before our top-level shutdown handler could run, producing zombie tasks. Fixed by disabling Uvicorn's signal handlers (`server.install_signal_handlers = lambda: None`) and managing shutdown explicitly at the asyncio level.

**Non-blocking camera capture.** `picamera2.capture_file()` is synchronous and takes ~200 ms. Calling it directly from the radar event handler would block the asyncio loop and stall the 10 Hz radar pipeline. Captures are offloaded to a worker thread via `asyncio.to_thread()`, keeping the event loop responsive.

**Cross-machine development.** The camera driver lazy-imports `picamera2` inside its `open()` method, so the module remains importable on the development Mac (where `picamera2` does not exist). Only an explicit `open()` call would fail. This keeps the same code runnable everywhere for static analysis, tests, and IDE tooling.

**Staring array vs rotating radar.** For full 360° coverage, the chosen strategy is multiple fixed LD2450 units rather than a mechanically rotating sensor — no moving parts, no rotation latency, continuous detection. This mirrors how modern radar arrays (AESA, modern air defense) work. Validating multi-radar 24 GHz coexistence (potential mutual interference) is a planned milestone.

---

## Repository Structure

```
project-sentinel/
├── sentinel/
│   ├── core/           # Domain models, EventBus, persistence, queries
│   ├── drivers/        # LD2450 parser/simulator/UART driver, Pi Camera driver
│   ├── consumers/      # Event subscribers (currently: CameraTrigger)
│   ├── api/            # FastAPI app + WebSocket endpoints
│   └── main.py         # Pipeline orchestration
├── web/                # Tactical dashboard (HTML/CSS/JS)
├── ml/                 # PyTorch training pipeline (dataset, model, train loop)
├── tests/              # Pytest suite
├── config/             # YAML configuration (default + local override)
└── docs/               # Architecture, project journal
```

---

## Roadmap

- [ ] Complete dataset construction (FiftyOne + Open Images / COCO / Places)
- [ ] First CNN training run on M-series MPS backend
- [ ] Transfer learning comparison (MobileNetV3)
- [ ] Hailo NPU model compilation
- [ ] Pan/tilt servo control with slew-to-cue actuation
- [ ] Mobile-responsive dashboard
- [ ] Multi-radar 360° coverage with interference validation
- [ ] systemd service for production auto-start
- [ ] Protocol documentation (`docs/ld2450-protocol.md`)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built by **Léo R.** — engineering student (MPSI), targeting embedded systems and defense / aerospace.

This project is developed as a learning vehicle and portfolio piece. Feedback and questions welcome through GitHub issues.
