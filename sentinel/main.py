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

import argparse
import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

import uvicorn

from sentinel.api.app import create_app
from sentinel.config import RadarSourceType, SentinelConfig, load_config
from sentinel.core.event_bus import EventBus
from sentinel.core.models import RadarFrame
from sentinel.core.persistence import DetectionLogger
from sentinel.drivers.ld2450_parser import LD2450ParseError, parse_frame
from sentinel.drivers.ld2450_simulator import LD2450Simulator

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
# CONSTRUCTION DE LA SOURCE RADAR (selon la config)
# ─────────────────────────────────────────────────────────────────────────────

def build_radar_source(config: SentinelConfig):
    """
    Construit la source radar en fonction de la configuration.

    Retourne un objet exposant une méthode async stream() qui yield des
    trames binaires (bytes). Pour l'instant, seul le simulateur est
    supporté. Le vrai LD2450Driver sera ajouté à l'étape suivante.

    Cette fonction est le point d'inversion de dépendance entre le pipeline
    et le hardware : tout le reste du code consomme l'interface "qui émet
    des trames binaires" sans savoir si c'est simulé ou réel.
    """
    if config.radar.source == RadarSourceType.SIMULATOR:
        logger.info(
            "Source radar : SIMULATEUR (%d cibles aléatoires)",
            config.radar.simulator_targets,
        )
        sim = LD2450Simulator()
        sim.populate_random(n=config.radar.simulator_targets)
        return sim

    elif config.radar.source == RadarSourceType.UART:
        logger.info(
            "Source radar : UART (port=%s, baudrate=%d)",
            config.radar.uart_port,
            config.radar.uart_baudrate,
        )
        # Import lazy : pyserial-asyncio n'est requis qu'en mode UART.
        from sentinel.drivers.ld2450_driver import LD2450Driver
        return LD2450Driver(
            port=config.radar.uart_port,
            baudrate=config.radar.uart_baudrate,
        )

    else:
        raise ValueError(f"Source radar inconnue : {config.radar.source}")
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


async def stats_reporter(
    stats: FrameStats,
    shutdown_event: asyncio.Event,
    interval_seconds: float = 5.0,) -> None:
    """Log périodique des stats."""
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
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


async def web_server(
    bus: EventBus,
    shutdown_event: asyncio.Event,
    config: SentinelConfig,
) -> None:
    """
    Lance le serveur FastAPI/uvicorn avec la config fournie.

    Note: on désactive la gestion native des signaux d'uvicorn
    (install_signal_handlers=False) parce qu'elle entre en conflit
    avec notre propre gestion dans cli(). Sentinel gère SIGINT/SIGTERM
    au niveau asyncio top-level, uvicorn n'a pas à s'en mêler.
    """
    app = create_app(bus, db_path=config.persistence.db_path)
    uvicorn_config = uvicorn.Config(
        app,
        host=config.web.host,
        port=config.web.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)

    # Important: empêche uvicorn d'installer ses propres signal handlers.
    # Sans ça, uvicorn intercepte SIGINT/SIGTERM et notre shutdown_event
    # n'est jamais déclenché.
    server.install_signal_handlers = lambda: None

    logger.info(
        " --- Serveur web démarré sur http://%s:%d",
        "localhost" if config.web.host == "0.0.0.0" else config.web.host,
        config.web.port,
    )

    server_task = asyncio.create_task(server.serve())

    # Attendre le shutdown signal venu d'ailleurs.
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass

    logger.info("Arrêt du serveur web demandé")
    server.should_exit = True

    # Donner 2 secondes max à uvicorn pour fermer proprement,
    # puis on force.
    try:
        await asyncio.wait_for(server_task, timeout=2.0)
    except asyncio.TimeoutError:
        logger.warning("Uvicorn n'a pas fermé en 2s, annulation forcée")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# AMORÇAGE
# ─────────────────────────────────────────────────────────────────────────────

async def main(config: SentinelConfig) -> None:
    """
    Construit les composants depuis la config et lance les coroutines.

    Toutes les valeurs (port, source radar, chemin DB, etc.) proviennent
    de la config — aucun hardcode ici. Cela permet de faire varier le
    comportement de Sentinel sans toucher au code Python.
    """
    bus = EventBus()

    # Construction de la source radar selon la config.
    radar_source = build_radar_source(config)

    # Subscribers
    stats = FrameStats()
    bus.subscribe(EVENT_RADAR_FRAME, stats.on_frame)
    bus.subscribe(EVENT_RADAR_FRAME, log_frame_to_console)
    bus.subscribe(EVENT_RADAR_PARSE_ERROR, stats.on_parse_error)

    # Persistance SQLite (optionnelle selon la config).
    detection_logger: DetectionLogger | None = None
    if config.persistence.enabled:
        detection_logger = DetectionLogger(db_path=config.persistence.db_path)
        await detection_logger.start(source=config.radar.source.value)
        bus.subscribe(EVENT_RADAR_FRAME, detection_logger.on_frame)
    else:
        logger.info("Persistance SQLite désactivée par configuration")

    # Mécanisme d'arrêt propre.
    shutdown_event = asyncio.Event()

    def request_shutdown() -> None:
        logger.info("Signal d'arrêt reçu, fermeture en cours...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_shutdown)

    logger.info("🚀 Sentinel démarré — Ctrl+C pour arrêter")
    logger.info(
        "📺 Dashboard : http://%s:%d",
        "localhost" if config.web.host == "0.0.0.0" else config.web.host,
        config.web.port,
    )

    # Création des tâches individuelles pour pouvoir les annuler proprement
    # en cas de shutdown lent.
    pipeline_task = asyncio.create_task(
        radar_pipeline(radar_source, bus, shutdown_event)
    )
    stats_task = asyncio.create_task(
        stats_reporter(stats, shutdown_event, config.logging.stats_interval_seconds)
    )
    web_task = asyncio.create_task(
        web_server(bus, shutdown_event, config)
    )

    all_tasks = [pipeline_task, stats_task, web_task]

    # On attend soit que toutes les tâches terminent naturellement (improbable),
    # soit que le shutdown_event soit déclenché.
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        # SIGINT/SIGTERM reçu avant même que shutdown_event soit déclenché
        # (cas rare mais possible). On déclenche manuellement.
        shutdown_event.set()

    logger.info("Arrêt des tâches en cours...")

    # On laisse 3 secondes aux tâches pour s'arrêter proprement, puis on
    # les annule de force.
    for task in all_tasks:
        if not task.done():
            task.cancel()

    # Attente finale, avec absorption des CancelledError.
    await asyncio.gather(*all_tasks, return_exceptions=True)

    # Clôture propre du logger SQLite (si actif).
    if detection_logger is not None:
        await detection_logger.stop()

    final_snap = stats.snapshot()
    logger.info("=" * 60)
    logger.info("Sentinel arrêté. Stats finales :")
    for key, value in final_snap.items():
        logger.info("  %s: %s", key, value)
    logger.info("Bus : %s", bus.stats)
    if detection_logger is not None:
        logger.info("SQLite : %s", detection_logger.stats)
        
# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────────────────────

def cli() -> None:
    """
    Point d'entrée CLI : parse les arguments, charge la config, lance main().

    Usage :
        python -m sentinel.main                       (config par défaut)
        python -m sentinel.main --config FILE.yaml    (config explicite)
        python -m sentinel.main --log-level DEBUG     (override du niveau de log)
    """
    parser = argparse.ArgumentParser(
        description="Sentinel — Système de surveillance radar autonome",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Chemin vers un fichier YAML de config autonome",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override du niveau de log (défaut: lu depuis la config)",
    )
    args = parser.parse_args()

    # Chargement de la config
    config = load_config(config_file=args.config)

    # Override CLI du niveau de log s'il est fourni
    log_level = args.log_level or config.logging.level

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(main(config))


if __name__ == "__main__":
    cli()