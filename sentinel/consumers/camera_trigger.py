"""
CameraTrigger — déclenche une capture caméra sur détection radar.

Ce module implémente le cœur du « slew-to-cue » logiciel de Sentinel : quand
le radar détecte une cible, la caméra prend une photo. Le CameraTrigger est un
*consumer* (abonné à l'EventBus) : il ne produit rien, il réagit aux trames
radar publiées par le pipeline.

Deux problèmes sont gérés ici :

1. THROTTLE (anti-saturation). Le radar émet à 10 Hz. Sans limite, on
   capturerait 10 photos/seconde tant qu'une cible est présente → disque
   saturé et caméra incapable de suivre. On impose donc un délai minimum
   entre deux captures (cooldown).

2. CAPTURE NON-BLOQUANTE. picamera2.capture_file() est synchrone et prend
   ~150-300 ms. Appelé directement dans la boucle asyncio, il bloquerait tout
   le pipeline (le radar décrocherait, le WebSocket laggerait). On l'exécute
   donc dans un thread via asyncio.to_thread(), ce qui rend la main à la
   boucle asyncio pendant que la photo s'écrit.

Le nom de fichier encode la position radar de la cible (slot, distance, angle),
ce qui crée un lien explicite entre la détection et l'image — précieux pour
constituer le futur dataset d'entraînement IA.

Auteur : Léo
Projet : Sentinel — Système de surveillance autonome
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from sentinel.core.models import RadarFrame, Target
from sentinel.drivers.camera import CameraError, SentinelCamera

logger = logging.getLogger(__name__)


# Délai minimum (secondes) entre deux captures. Surchargé depuis la config.
DEFAULT_COOLDOWN_SECONDS: float = 5.0


class CameraTrigger:
    """
    Abonné EventBus qui capture une photo quand le radar détecte une cible.

    Cycle de vie aligné sur celui de la caméra : start() ouvre la caméra,
    stop() la ferme. Entre les deux, on_frame() est appelé par le bus à chaque
    trame radar et décide (throttle) de capturer ou non.
    """

    def __init__(
        self,
        camera: Optional[SentinelCamera] = None,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        """
        Args:
            camera: instance de SentinelCamera à utiliser. Si None, une instance
                    par défaut est créée (résolution et dossier par défaut).
            cooldown_seconds: délai minimum entre deux captures.
        """
        self._camera = camera if camera is not None else SentinelCamera()
        self._cooldown = cooldown_seconds

        # Horodatage monotone de la dernière capture. -inf = jamais capturé,
        # donc la première cible déclenche immédiatement une photo.
        self._last_capture_t: float = float("-inf")

        # Garde-fou anti-chevauchement : si une capture est déjà en cours dans
        # son thread, on ne lance pas une seconde capture en parallèle (la
        # caméra ne supporte pas deux capture_file() simultanés).
        self._capture_in_progress: bool = False

        # Compteurs pour monitoring / stats finales.
        self.captures_done: int = 0
        self.captures_skipped_cooldown: int = 0

        self._started: bool = False

    # ─── Cycle de vie ────────────────────────────────────────────────────

    def start(self) -> None:
        """Ouvre la caméra. À appeler une fois avant de s'abonner au bus."""
        self._camera.open()
        self._started = True
        logger.info(
            "CameraTrigger démarré (cooldown=%.1fs)", self._cooldown
        )

    def stop(self) -> None:
        """Ferme la caméra. Idempotent."""
        if self._started:
            self._camera.close()
            self._started = False
            logger.info(
                "CameraTrigger arrêté — %d capture(s), %d ignorée(s) (cooldown)",
                self.captures_done,
                self.captures_skipped_cooldown,
            )

    # ─── Abonné EventBus ─────────────────────────────────────────────────

    async def on_frame(self, frame: RadarFrame) -> None:
        """
        Appelé par l'EventBus à chaque trame radar.

        Ne capture que si : la trame contient au moins une cible, le cooldown
        est écoulé, et aucune capture n'est déjà en cours.
        """
        if not frame.has_targets:
            return

        # Throttle : a-t-on attendu assez longtemps depuis la dernière capture ?
        now = time.monotonic()
        if now - self._last_capture_t < self._cooldown:
            self.captures_skipped_cooldown += 1
            return

        # Anti-chevauchement : une capture est-elle déjà en cours ?
        if self._capture_in_progress:
            return

        # On choisit la cible la plus proche comme sujet principal de la photo.
        primary = min(frame.targets, key=lambda t: t.distance_m)

        # On verrouille AVANT de lancer le thread, et on met à jour le timestamp
        # tout de suite : ainsi les trames suivantes (10 Hz) sont throttlées même
        # pendant que la capture est en cours.
        self._capture_in_progress = True
        self._last_capture_t = now

        # Lancement de la capture dans un thread pour ne pas bloquer asyncio.
        await self._capture_async(primary)

    async def _capture_async(self, target: Target) -> None:
        """Exécute la capture bloquante dans un thread, sans bloquer la boucle."""
        # Étiquette encodant la position radar : ex. "T0_d2.3m_a-15deg".
        label = (
            f"T{target.slot}"
            f"_d{target.distance_m:.1f}m"
            f"_a{target.angle_degrees:+.0f}deg"
        )
        try:
            path = await asyncio.to_thread(self._camera.capture, label)
            self.captures_done += 1
            logger.info("📸 Capture déclenchée par radar : %s", path.name)
        except CameraError as exc:
            logger.error("Échec de la capture caméra : %s", exc)
        finally:
            self._capture_in_progress = False
