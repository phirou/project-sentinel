"""
Point d'entrée principal de Sentinel — orchestration du pipeline radar
et du serveur web.

Ce module assemble :
    - Source radar (simulateur ou vrai LD2450)
    - Parser binaire
    - EventBus pour le dispatch des événements
    - Subscribers (logger console, stats)
    - Serveur FastAPI + WebSocket pour le dashboard

Le tout tourne en asyncio sur une boucle unique. Le simulateur émet
des trames, le parser les décode, le bus dispatche, et le navigateur
reçoit en temps réel via WebSocket.

Lancement :
    python -m sentinel.main

Dashboard accessible ensuite sur :
    http://localhost:8000

Auteur : Léo
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

import uvicorn

from sentinel.api.app import create_app
from sentinel.core.event_bus import EventBus
from sentinel.core.models import RadarFrame
from sentinel.drivers.ld2450_parser import LD2450ParseError, parse_frame
from sentinel.drivers.ld2450_simulator import LD2450Simulator
from sentinel.core.persistence import DetectionLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.main")

EVENT_RADAR_FRAME = "radar.frame"
EVENT_RADAR_PARSE_ERROR = "radar.parse_error"


# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIBERS DU BUS (inchangés depuis l'étape 6)
# ─────────────────────────────────────────────────────────────────────────────

class FrameStats:
    """Compteur d'événements pour monitoring."""

    def __init__(self) -> None:
        self.frames_received: int = 0
        self.frames_with_targets: int = 0
        self.parse_errors: int = 0
        self.total_targets: int = 0

    async def on_frame(self, frame: RadarFrame) -> None:
        self.frames_received += 1
        self.total_targets += frame.target_count
        if frame.has_targets:
            self.frames_with_targets += 1

    async def on_parse_error(self, error: Exception) -> None:
        self.parse_errors += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "frames_received": self.frames_received,
            "frames_with_targets": self.frames_with_targets,
            "parse_errors": self.parse_errors,
            "total_targets": self.total_targets,
            "avg_targets_per_frame": (
                self.total_targets / self.frames_received
                if self.frames_received > 0
                else 0.0
            ),
        }


async def log_frame_to_console(frame: RadarFrame) -> None:
    """Subscriber console — n'affiche que les trames avec cibles."""
    if not frame.has_targets:
        return
    targets_str = " | ".join(
        f"T{t.slot}: {t.distance_m:4.1f}m {t.angle_degrees:+5.1f}° "
        f"{t.speed_ms:+5.2f}m/s [{t.state.value}]"
        for t in frame.targets
    )
    logger.info("📡 %d cible(s) — %s", frame.target_count, targets_str)


# ─────────────────────────────────────────────────────────────────────────────
# COROUTINES DU PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def radar_pipeline(
    simulator: LD2450Simulator,
    bus: EventBus,
    shutdown_event: asyncio.Event,
) -> None:
    """Pipeline : simulateur → parser → bus."""
    logger.info("Démarrage du pipeline radar (source: simulateur)")
    async for frame_bytes in simulator.stream():
        if shutdown_event.is_set():
            logger.info("Arrêt du pipeline radar demandé")
            break
        try:
            frame = parse_frame(frame_bytes)
        except LD2450ParseError as exc:
            logger.warning("Erreur de parsing : %s", exc)
            await bus.publish(EVENT_RADAR_PARSE_ERROR, exc)
            continue
        await bus.publish(EVENT_RADAR_FRAME, frame)


async def stats_reporter(stats: FrameStats, shutdown_event: asyncio.Event) -> None:
    """Log périodique des stats toutes les 5 secondes."""
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
            break
        except asyncio.TimeoutError:
            snap = stats.snapshot()
            logger.info(
                "📊 Stats : %d trames | %d avec cibles | %d erreurs | %.2f cibles/trame",
                snap["frames_received"],
                snap["frames_with_targets"],
                snap["parse_errors"],
                snap["avg_targets_per_frame"],
            )


async def web_server(bus: EventBus, shutdown_event: asyncio.Event) -> None:
    """
    Coroutine qui lance le serveur web FastAPI/uvicorn.

    On utilise uvicorn.Server programmatically (et non en CLI) pour qu'il
    tourne sur la même boucle asyncio que le reste du pipeline. Ça évite
    les threads et garantit que les handlers WebSocket voient les mêmes
    événements que les autres subscribers du bus.
    """
    app = create_app(bus)
    config = uvicorn.Config(
        app,
        host="0.0.0.0",   # accessible depuis le réseau local (pour démo)
        port=8000,
        log_level="warning",  # uvicorn log seulement les warnings/errors
        access_log=False,     # pas de log d'accès HTTP, trop verbeux
    )
    server = uvicorn.Server(config)

    logger.info("🌐 Serveur web démarré sur http://localhost:8000")

    # On lance le serveur dans une tâche pour pouvoir l'arrêter proprement.
    server_task = asyncio.create_task(server.serve())

    # On attend soit le shutdown global, soit la fin du serveur (improbable).
    await shutdown_event.wait()

    logger.info("Arrêt du serveur web demandé")
    server.should_exit = True
    await server_task


# ─────────────────────────────────────────────────────────────────────────────
# AMORÇAGE
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Construit les composants et lance les coroutines en parallèle."""
    bus = EventBus()
    simulator = LD2450Simulator()
    simulator.populate_random(n=2)

    stats = FrameStats()
    bus.subscribe(EVENT_RADAR_FRAME, stats.on_frame)
    bus.subscribe(EVENT_RADAR_FRAME, log_frame_to_console)
    bus.subscribe(EVENT_RADAR_PARSE_ERROR, stats.on_parse_error)

    # Persistance SQLite : abonnement au bus, écriture asynchrone non bloquante.
    detection_logger = DetectionLogger(db_path="data/sentinel.db")
    await detection_logger.start(source="simulator")
    bus.subscribe(EVENT_RADAR_FRAME, detection_logger.on_frame)

    shutdown_event = asyncio.Event()

    def request_shutdown() -> None:
        logger.info("Signal d'arrêt reçu, fermeture en cours...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_shutdown)

    logger.info("🚀 Sentinel démarré — Ctrl+C pour arrêter")
    logger.info("📺 Dashboard : http://localhost:8000")

    try:
        await asyncio.gather(
            radar_pipeline(simulator, bus, shutdown_event),
            stats_reporter(stats, shutdown_event),
            web_server(bus, shutdown_event),
        )
    except asyncio.CancelledError:
        pass

    # Clôture propre du logger SQLite.
    await detection_logger.stop()

    final_snap = stats.snapshot()
    logger.info("=" * 60)
    logger.info("Sentinel arrêté. Stats finales :")
    for key, value in final_snap.items():
        logger.info("  %s: %s", key, value)
    logger.info("Bus : %s", bus.stats)
    logger.info("SQLite : %s", detection_logger.stats)


if __name__ == "__main__":
    asyncio.run(main())