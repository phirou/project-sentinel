"""
Driver UART asynchrone pour le capteur radar HLK-LD2450.

Lit le flux binaire continu émis par le LD2450 sur son port série
et le découpe en trames de 30 octets prêtes à être parsées.

Architecture :
    - Lecture non-bloquante via pyserial-asyncio (intégration native asyncio)
    - Synchronisation sur le header AA FF 03 00 (gestion des trames partielles
      et des octets parasites)
    - Validation du footer 55 CC pour rejeter les faux positifs
    - Reconnexion automatique en cas de perte de port (utile en production)

Ce driver expose la MÊME interface que LD2450Simulator (méthode async stream()
qui yield des bytes) — c'est l'inversion de dépendance qui permet au pipeline
amont de ne pas savoir s'il consomme du simulé ou du réel.

Auteur : Léo
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import serial_asyncio

from sentinel.drivers.ld2450_parser import (
    FRAME_FOOTER,
    FRAME_HEADER,
    FRAME_SIZE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Vitesse de transmission UART du LD2450 (constructeur Hi-Link).
DEFAULT_BAUDRATE: int = 256000

# Taille max du buffer de réception avant qu'on le considère anormal.
# 10 trames non synchronisées = 300 octets = signe d'un problème (mauvais baudrate ?).
MAX_BUFFER_SIZE: int = 300

# Délai entre tentatives de reconnexion si le port disparaît.
RECONNECT_DELAY_S: float = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class LD2450Driver:
    """
    Driver série pour LD2450 réel.

    Usage typique :
        driver = LD2450Driver(port="/dev/ttyAMA0", baudrate=256000)
        async for frame_bytes in driver.stream():
            frame = parse_frame(frame_bytes)
            ...

    Le driver gère :
        - L'ouverture/fermeture du port
        - La reconnexion automatique en cas de perte
        - La synchronisation header → footer
        - La fragmentation et la concaténation des lectures
    """

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> None:
        self.port = port
        self.baudrate = baudrate

        # Compteurs internes pour monitoring/debug.
        self._frames_emitted: int = 0
        self._invalid_footers: int = 0
        self._reconnect_count: int = 0

    # ─── API publique ───────────────────────────────────────────────

    async def stream(self) -> AsyncIterator[bytes]:
        """
        Itère sur les trames de 30 octets reçues du LD2450.

        Cette méthode tourne en boucle infinie : si le port se déconnecte,
        elle tente de se reconnecter après RECONNECT_DELAY_S secondes.

        Yields:
            Des bytes de longueur exactement FRAME_SIZE (30), validés en
            header ET en footer.
        """
        while True:
            try:
                async for frame in self._read_loop():
                    yield frame
            except (OSError, serial_asyncio.serial.SerialException) as exc:
                logger.warning(
                    "Port série déconnecté ou inaccessible (%s) — "
                    "nouvelle tentative dans %.1fs",
                    exc,
                    RECONNECT_DELAY_S,
                )
                self._reconnect_count += 1
                await asyncio.sleep(RECONNECT_DELAY_S)

    # ─── Lecture brute du port ──────────────────────────────────────

    async def _read_loop(self) -> AsyncIterator[bytes]:
        """
        Boucle de lecture interne, ouvre le port et yield des trames.

        Cette coroutine ne capture pas les exceptions de bas niveau :
        elle les laisse remonter vers stream() qui gère la reconnexion.
        """
        logger.info(
            "Ouverture du port série %s à %d bauds", self.port, self.baudrate
        )

        # open_serial_connection retourne un couple (reader, writer).
        # Le writer n'est pas utilisé ici (on n'envoie rien au radar pour
        # l'instant — la configuration radar pourrait être un ajout futur).
        reader, writer = await serial_asyncio.open_serial_connection(
            url=self.port,
            baudrate=self.baudrate,
        )

        try:
            # Buffer de bytes accumulés depuis le port.
            # On garde le buffer entre les itérations parce qu'une lecture
            # peut renvoyer N octets dont seulement une partie forme une
            # trame complète — le reste sera utile pour la prochaine.
            buffer = bytearray()

            while True:
                # Lecture non-bloquante : on lit jusqu'à 64 octets.
                # Pourquoi 64 ? C'est une lecture "raisonnable" : ni trop
                # gros (latence ajoutée si peu de données), ni trop petit
                # (trop d'appels système).
                chunk = await reader.read(64)
                if not chunk:
                    # Le reader est fermé : exception remontée à stream().
                    raise OSError("Port série fermé par le système")

                buffer.extend(chunk)

                # On essaie d'extraire toutes les trames complètes du buffer.
                # On boucle parce qu'un chunk peut contenir plusieurs trames.
                while True:
                    frame = self._try_extract_frame(buffer)
                    if frame is None:
                        # Plus de trame complète dans le buffer, on attend
                        # plus de données du port.
                        break
                    self._frames_emitted += 1
                    yield bytes(frame)

                # Sécurité : si le buffer devient anormalement gros, on le
                # vide. Symptôme typique : mauvais baudrate, le radar parle
                # mais on ne reconnaît jamais de header.
                if len(buffer) > MAX_BUFFER_SIZE:
                    logger.warning(
                        "Buffer série anormalement gros (%d octets) — reset. "
                        "Le baudrate est-il correct ?",
                        len(buffer),
                    )
                    buffer.clear()

        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Port série %s fermé", self.port)

    # ─── Synchronisation header/footer ──────────────────────────────

    def _try_extract_frame(self, buffer: bytearray) -> Optional[bytearray]:
        """
        Tente d'extraire une trame valide depuis le buffer.

        Cherche le header dans le buffer. Si une trame complète et valide
        est trouvée :
            - La trame est retournée
            - Le buffer est nettoyé jusqu'à la fin de la trame extraite

        Sinon retourne None et nettoie le buffer du début inutile.

        Returns:
            Un bytearray de FRAME_SIZE octets si trame valide trouvée,
            None si pas encore assez de données ou pas de header en vue.
        """
        # Recherche du header dans le buffer.
        # bytes.find() est très rapide (implémenté en C), pas besoin d'optim.
        header_index = buffer.find(FRAME_HEADER)

        if header_index == -1:
            # Pas de header trouvé. On peut jeter presque tout le buffer,
            # SAUF les 3 derniers octets : le header pourrait commencer
            # dans les derniers octets et se compléter au prochain chunk.
            if len(buffer) > len(FRAME_HEADER):
                del buffer[: -len(FRAME_HEADER) + 1]
            return None

        # On a trouvé un header. On jette tout ce qui est avant lui (bruit
        # ou résidu d'une trame précédente corrompue).
        if header_index > 0:
            logger.debug(
                "Resync : %d octets parasites avant header, jetés", header_index
            )
            del buffer[:header_index]

        # Le buffer commence maintenant par le header. On vérifie qu'on a
        # assez d'octets pour une trame complète.
        if len(buffer) < FRAME_SIZE:
            return None

        # On a un candidat. On vérifie le footer.
        candidate = buffer[:FRAME_SIZE]
        if candidate[-len(FRAME_FOOTER):] != FRAME_FOOTER:
            # Footer incorrect : c'était un faux positif (le pattern AA FF 03 00
            # apparaissait par hasard dans des données). On avance d'1 octet
            # pour chercher un autre header plus loin.
            self._invalid_footers += 1
            logger.debug(
                "Header trouvé mais footer invalide (faux positif) — resync"
            )
            del buffer[:1]
            return None

        # Trame valide : on la retire du buffer et on la retourne.
        frame = bytearray(candidate)
        del buffer[:FRAME_SIZE]
        return frame

    # ─── Introspection ──────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Statistiques du driver pour monitoring."""
        return {
            "frames_emitted": self._frames_emitted,
            "invalid_footers": self._invalid_footers,
            "reconnect_count": self._reconnect_count,
        }