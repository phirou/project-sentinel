"""
Pilote de la caméra Sentinel (Raspberry Pi Camera Module 3 / capteur IMX708).

Ce module encapsule la capture d'images via la bibliothèque picamera2 (la
couche Python officielle au-dessus de libcamera sur Raspberry Pi). Il expose
une interface simple et synchrone-friendly, conçue pour être appelée plus tard
depuis le pipeline asyncio de Sentinel sur événement de détection radar
(principe « slew-to-cue » : le radar désigne, la caméra capture).

Choix de conception :
- On garde la caméra OUVERTE entre les captures (start/stop coûteux ~300 ms).
  La classe est donc un gestionnaire de cycle de vie : open() une fois,
  capture() autant de fois que voulu, close() à la fin.
- La capture elle-même est bloquante (picamera2 n'est pas asyncio). Pour ne pas
  bloquer la boucle asyncio plus tard, on l'exécutera dans un thread via
  asyncio.to_thread() — mais ce module reste agnostique de ça.
- Les images sont horodatées en UTC pour cohérence avec la persistance SQLite.

Usage typique (test autonome) :
    cam = SentinelCamera()
    cam.open()
    chemin = cam.capture()        # capture vers un fichier horodaté
    cam.close()

Auteur : Léo
Projet : Sentinel — Système de surveillance autonome
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Résolution de capture par défaut. On reste en HD (1280x720) plutôt qu'en
# pleine résolution (4608x2592) : c'est largement suffisant pour la
# classification IA, et bien plus léger à traiter sur le Pi et le NPU Hailo.
DEFAULT_WIDTH: int = 1280
DEFAULT_HEIGHT: int = 720

# Répertoire où sont stockées les captures. Relatif à la racine du projet.
DEFAULT_CAPTURE_DIR: Path = Path("data/captures")


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class CameraError(Exception):
    """Levée quand une opération caméra échoue (ouverture, capture)."""


# ─────────────────────────────────────────────────────────────────────────────
# PILOTE CAMÉRA
# ─────────────────────────────────────────────────────────────────────────────

class SentinelCamera:
    """
    Encapsule la Pi Camera Module 3 pour la capture d'images à la demande.

    La caméra est ouverte une fois puis réutilisée pour chaque capture, ce qui
    évite le coût de réinitialisation du capteur (~300 ms) à chaque photo.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        capture_dir: Path = DEFAULT_CAPTURE_DIR,
    ) -> None:
        """
        Args:
            width: largeur de capture en pixels.
            height: hauteur de capture en pixels.
            capture_dir: répertoire de sortie des images (créé si absent).
        """
        self._width = width
        self._height = height
        self._capture_dir = Path(capture_dir)

        # L'objet Picamera2 sous-jacent. None tant que open() n'a pas été appelé.
        # On l'importe paresseusement dans open() pour que ce module soit
        # importable même sur une machine sans picamera2 (ex : le Mac de dev).
        self._picam2: Optional[object] = None
        self._is_open: bool = False

    # ─── Cycle de vie ────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Initialise et démarre la caméra.

        Import paresseux de picamera2 : ainsi, importer SentinelCamera sur une
        machine sans la lib (le Mac) ne plante pas — seul open() échouera.

        Raises:
            CameraError: si picamera2 est indisponible ou si le démarrage échoue.
        """
        if self._is_open:
            logger.warning("open() appelé alors que la caméra est déjà ouverte")
            return

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise CameraError(
                "picamera2 introuvable. Ce module nécessite un Raspberry Pi "
                "avec picamera2 (venv créé avec --system-site-packages)."
            ) from exc

        try:
            self._picam2 = Picamera2()
            # Configuration « still » (photo) à la résolution voulue.
            config = self._picam2.create_still_configuration(
                main={"size": (self._width, self._height)}
            )
            self._picam2.configure(config)
            self._picam2.start()
            self._is_open = True
            logger.info(
                "Caméra ouverte (%dx%d)", self._width, self._height
            )
        except Exception as exc:  # noqa: BLE001 — on remonte tout en CameraError
            raise CameraError(f"Échec de l'ouverture de la caméra : {exc}") from exc

    def close(self) -> None:
        """Arrête et libère la caméra. Idempotent (sûr à appeler plusieurs fois)."""
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Erreur lors de la fermeture de la caméra : %s", exc)
            finally:
                self._picam2 = None
                self._is_open = False
                logger.info("Caméra fermée")

    # ─── Capture ─────────────────────────────────────────────────────────

    def capture(self, label: Optional[str] = None) -> Path:
        """
        Capture une image et l'enregistre dans capture_dir.

        Le nom de fichier est horodaté (UTC) pour être unique et triable :
            capture_2026-05-22T21-30-05_123456.jpg
        Un label optionnel (ex : « radar_T0 ») est inséré dans le nom pour
        relier la photo à l'événement qui l'a déclenchée.

        Args:
            label: étiquette optionnelle insérée dans le nom de fichier.

        Returns:
            Le chemin du fichier image créé.

        Raises:
            CameraError: si la caméra n'est pas ouverte ou si la capture échoue.
        """
        if not self._is_open or self._picam2 is None:
            raise CameraError("capture() appelé alors que la caméra est fermée")

        # Crée le répertoire de sortie si nécessaire.
        self._capture_dir.mkdir(parents=True, exist_ok=True)

        # Construit un nom de fichier horodaté et unique.
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y-%m-%dT%H-%M-%S_%f")
        suffix = f"_{label}" if label else ""
        filepath = self._capture_dir / f"capture_{stamp}{suffix}.jpg"

        try:
            # capture_file() est bloquant : il déclenche la prise et écrit le JPEG.
            self._picam2.capture_file(str(filepath))
            logger.info("Image capturée : %s", filepath)
            return filepath
        except Exception as exc:  # noqa: BLE001
            raise CameraError(f"Échec de la capture : {exc}") from exc

    # ─── Context manager ─────────────────────────────────────────────────

    def __enter__(self) -> "SentinelCamera":
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# TEST AUTONOME
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    """
    Point d'entrée de test : capture 3 images espacées d'une seconde.

    Lancement :
        python -m sentinel.drivers.camera
    """
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Test autonome de la caméra Sentinel — 3 captures")

    with SentinelCamera() as cam:
        for i in range(3):
            path = cam.capture(label=f"test{i}")
            print(f"  → capture {i + 1}/3 : {path}")
            time.sleep(1.0)

    logger.info("Test terminé. Images dans %s", DEFAULT_CAPTURE_DIR)


if __name__ == "__main__":
    _main()
