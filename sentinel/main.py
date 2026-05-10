"""
Point d'entrée principal de Sentinel — orchestration du pipeline radar.

Ce module assemble les briques fondamentales :
    - Source radar (simulateur ou vrai LD2450)
    - Parser binaire
    - EventBus pour le dispatch des événements
    - Subscribers (logger console, stats, ...)

Le pipeline tourne en asyncio et émet des événements 'radar.frame' à
chaque trame parsée. Tout consommateur en aval (dashboard, persistance,
tracking, IA) s'abonne au bus et travaille de manière indépendante.

Lancement :
    python -m sentinel.main

Ce script utilise le simulateur par défaut. Quand le LD2450 physique
sera disponible, on remplacera RadarSource par le vrai driver UART
sans toucher au reste du code (inversion de dépendance).

Auteur : Léo
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from sentinel.core.event_bus import EventBus
from sentinel.core.models import RadarFrame
from sentinel.drivers.ld2450_parser import LD2450ParseError, parse_frame
from sentinel.drivers.ld2450_simulator import LD2450Simulator

# Configuration du logging : format clair, niveau INFO par défaut.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.main")


# ─────────────────────────────────────────────────────────────────────────────
# NOMS D'ÉVÉNEMENTS (constantes pour éviter les fautes de frappe)
# ─────────────────────────────────────────────────────────────────────────────

EVENT_RADAR_FRAME = "radar.frame"
EVENT_RADAR_PARSE_ERROR = "radar.parse_error"


# ─────────────────────────────────────────────────────────────────────────────
# SUBSCRIBERS DU BUS
# ─────────────────────────────────────────────────────────────────────────────

class FrameStats:
    """
    Compteur d'événements minimaliste, pour monitorer la santé du pipeline.

    S'abonne au bus et incrémente ses compteurs à chaque trame.
    À utiliser comme un singleton : une instance, abonnée une fois.
    """

    def __init__(self) -> None:
        self.frames_received: int = 0
        self.frames_with_targets: int = 0
        self.parse_errors: int = 0
        self.total_targets: int = 0

    async def on_frame(self, frame: RadarFrame) -> None:
        """Handler appelé à chaque événement 'radar.frame'."""
        self.frames_received += 1
        self.total_targets += frame.target_count
        if frame.has_targets:
            self.frames_with_targets += 1

    async def on_parse_error(self, error: Exception) -> None:
        """Handler appelé à chaque événement 'radar.parse_error'."""
        self.parse_errors += 1

    def snapshot(self) -> dict[str, Any]:
        """Retourne un instantané des compteurs (pour log périodique)."""
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
    """
    Subscriber simple qui affiche un résumé de chaque trame en console.

    On n'affiche que les trames avec au moins une cible pour ne pas
    saturer le terminal. À 10 Hz et sans cible visible, ça spamme vite.
    """
    if not frame.has_targets:
        return

    # Construction d'une ligne synthétique, lisible d'un coup d'œil.
    targets_str = " | ".join(
        f"T{t.slot}: {t.distance_m:4.1f}m {t.angle_degrees:+5.1f}° "
        f"{t.speed_ms:+5.2f}m/s [{t.state.value}]"
        for t in frame.targets
    )
    logger.info("📡 %d cible(s) — %s", frame.target_count, targets_str)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

async def radar_pipeline(
    simulator: LD2450Simulator,
    bus: EventBus,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Coroutine principale du pipeline radar.

    Lit le stream du simulateur en boucle, parse chaque trame, et publie
    les RadarFrame valides sur le bus. Les erreurs de parsing publient
    sur 'radar.parse_error'.

    S'arrête proprement quand shutdown_event est levé (Ctrl+C ou SIGTERM).
    """
    logger.info("Démarrage du pipeline radar (source: simulateur)")

    async for frame_bytes in simulator.stream():
        # Vérification de la demande d'arrêt entre chaque trame.
        if shutdown_event.is_set():
            logger.info("Arrêt du pipeline radar demandé")
            break

        try:
            frame = parse_frame(frame_bytes)
        except LD2450ParseError as exc:
            logger.warning("Erreur de parsing : %s", exc)
            await bus.publish(EVENT_RADAR_PARSE_ERROR, exc)
            continue

        # Publication asynchrone : tous les abonnés tournent en parallèle.
        await bus.publish(EVENT_RADAR_FRAME, frame)


async def stats_reporter(stats: FrameStats, shutdown_event: asyncio.Event) -> None:
    """
    Coroutine qui logue périodiquement les stats du pipeline.

    Tourne en parallèle du pipeline principal sur la même boucle asyncio
    grâce à asyncio.gather. Affiche un résumé toutes les 5 secondes.
    """
    while not shutdown_event.is_set():
        # asyncio.wait permet d'attendre soit le timeout soit le shutdown.
        # Plus propre que asyncio.sleep + check : sortie immédiate sur Ctrl+C.
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
            # Si on arrive ici, c'est que shutdown_event a été levé.
            break
        except asyncio.TimeoutError:
            # Timeout normal : on log les stats et on continue.
            snap = stats.snapshot()
            logger.info(
                "📊 Stats : %d trames | %d avec cibles | %d erreurs | %.2f cibles/trame",
                snap["frames_received"],
                snap["frames_with_targets"],
                snap["parse_errors"],
                snap["avg_targets_per_frame"],
            )


# ─────────────────────────────────────────────────────────────────────────────
# AMORÇAGE ET LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """
    Fonction principale : assemble les composants et lance les coroutines.

    Cette fonction est l'équivalent du `main()` traditionnel : elle
    construit le simulateur, le bus, les subscribers, et lance le tout
    en parallèle via asyncio.gather.
    """
    # 1. Construction du bus (instance locale, pas le singleton global).
    bus = EventBus()

    # 2. Construction du simulateur avec quelques cibles.
    simulator = LD2450Simulator()
    simulator.populate_random(n=2)

    # 3. Création des subscribers et abonnement au bus.
    stats = FrameStats()
    bus.subscribe(EVENT_RADAR_FRAME, stats.on_frame)
    bus.subscribe(EVENT_RADAR_FRAME, log_frame_to_console)
    bus.subscribe(EVENT_RADAR_PARSE_ERROR, stats.on_parse_error)

    # 4. Mécanisme d'arrêt propre (Ctrl+C, SIGTERM).
    shutdown_event = asyncio.Event()

    def request_shutdown() -> None:
        logger.info("Signal d'arrêt reçu, fermeture en cours...")
        shutdown_event.set()

    # Sous Linux/macOS, on intercepte SIGINT (Ctrl+C) et SIGTERM proprement.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_shutdown)

    # 5. Lancement des coroutines en parallèle.
    logger.info("🚀 Sentinel démarré — Ctrl+C pour arrêter")
    try:
        await asyncio.gather(
            radar_pipeline(simulator, bus, shutdown_event),
            stats_reporter(stats, shutdown_event),
        )
    except asyncio.CancelledError:
        # Levé naturellement à l'arrêt, on l'absorbe silencieusement.
        pass

    # 6. Affichage des stats finales.
    final_snap = stats.snapshot()
    logger.info("=" * 60)
    logger.info("Sentinel arrêté. Stats finales :")
    for key, value in final_snap.items():
        logger.info("  %s: %s", key, value)
    logger.info("Bus : %s", bus.stats)


# Point d'entrée standard Python : exécuter ce fichier directement
# lance la coroutine main() dans une boucle asyncio.
if __name__ == "__main__":
    asyncio.run(main())