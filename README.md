# Project Sentinel

> **Autonomous detection-only surveillance system** inspired by [Anduril Sentry Towers](https://www.anduril.com/) — combining mmWave radar, computer vision, and embedded AI on a Raspberry Pi 5.

[![Status](https://img.shields.io/badge/status-Phase%201%20in%20development-orange)](#-development-phases)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Vision

Sentinel is a low-cost, fully autonomous **slew-to-cue** surveillance system: a 24 GHz mmWave radar continuously detects motion in its field of view, automatically slews a pan/tilt camera toward the target, captures an image, and runs on-device AI classification — all in under one second, all locally on a Raspberry Pi 5 with a Hailo-8 neural accelerator.

**Strictly detection-only. No weapons. No autonomous engagement.**

This project explores the architecture of modern defense surveillance systems (sensor fusion, real-time tactical display, edge AI) using consumer-grade hardware, as a learning vehicle for embedded systems and defense-tech engineering.

---

## System Architecture