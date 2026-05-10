"""
Couche de persistance asynchrone de Sentinel.

Ce module écrit les détections radar dans une base SQLite locale,
sans bloquer la boucle asyncio principale grâce à aiosqlite.

Schéma de la base (3 tables liées) :

    sessions       : un démarrage de Sentinel = une session
        (id, started_at, ended_at, source, notes)

    frames         : une ligne par trame radar reçue (~10 Hz)
        (id, session_id, timestamp, target_count, raw_hex)

    detections     : une ligne par cible individuelle dans chaque trame
        (id, frame_id, slot, x_mm, y_mm, speed_cms, distance_m,
         angle_deg, state, resolution_mm)

Usage :
    logger = DetectionLogger(db_path='data/sentinel.db')
    await logger.start(source='simulator')
    bus.subscribe('radar.frame', logger.on_frame)
    # ... pipeline tourne ...
    await logger.stop()

La persistance est best-effort : si l'écriture échoue, on log l'erreur
et on continue. Le pipeline temps réel ne doit JAMAIS être bloqué par
un problème de persistance.

Auteur : Léo
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from sentinel.core.models import RadarFrame

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMA SQL
# ─────────────────────────────────────────────────────────────────────────────

# Le schéma est défini comme une chaîne SQL exécutée au démarrage.
# Note : les CREATE TABLE IF NOT EXISTS sont idempotents — on peut les
# relancer à chaque démarrage sans risque, c'est pratique pour l'init.
SCHEMA_SQL = """
-- Une session = un démarrage de Sentinel.
-- Permet de regrouper toutes les détections d'un même "run" pour le replay.
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    source      TEXT    NOT NULL,
    notes       TEXT
);

-- Une trame = une lecture du capteur (typiquement 10 Hz).
-- target_count peut être 0 (radar a parlé mais ne voit rien).
CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    timestamp     TEXT    NOT NULL,
    target_count  INTEGER NOT NULL,
    raw_hex       TEXT
);
CREATE INDEX IF NOT EXISTS idx_frames_session    ON frames(session_id);
CREATE INDEX IF NOT EXISTS idx_frames_timestamp  ON frames(timestamp);

-- Une détection = une cible dans une trame.
-- C'est la table la plus volumineuse, à indexer correctement.
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id        INTEGER NOT NULL REFERENCES frames(id),
    slot            INTEGER NOT NULL,
    x_mm            INTEGER NOT NULL,
    y_mm            INTEGER NOT NULL,
    speed_cms       INTEGER NOT NULL,
    distance_m      REAL    NOT NULL,
    angle_deg       REAL    NOT NULL,
    state           TEXT    NOT NULL,
    resolution_mm   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_frame     ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_distance  ON detections(distance_m);
"""


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class DetectionLogger:
    """
    Subscriber du bus qui persiste chaque trame et ses détections en SQLite.

    Cycle de vie :
        1. __init__(db_path) : configuration, pas d'I/O
        2. await start(source) : ouvre la connexion, crée le schéma,
           ouvre une nouvelle session
        3. bus.subscribe('radar.frame', logger.on_frame) : abonnement
        4. await stop() : ferme la session courante (ended_at), ferme la connexion

    Cette classe n'est PAS un singleton : on peut avoir plusieurs loggers
    actifs (par exemple un sur disque local, un sur disque externe pour
    backup). Pour Phase 1 on n'en a qu'un.
    """

    def __init__(self, db_path: str | Path = "data/sentinel.db") -> None:
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None
        self._session_id: Optional[int] = None

        # Compteurs internes pour monitoring.
        self._frames_logged = 0
        self._detections_logged = 0
        self._write_errors = 0

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def start(self, source: str = "simulator", notes: str = "") -> None:
        """
        Ouvre la base, crée le schéma si besoin, démarre une nouvelle session.

        Args:
            source: identifiant de la source radar (ex: 'simulator',
                    'ld2450_uart', 'replay'). Stocké dans la session.
            notes: notes libres associées à la session (optionnel).
        """
        # Crée le dossier parent si besoin (data/ par défaut).
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Ouverture de la base SQLite : %s", self.db_path)
        self._connection = await aiosqlite.connect(str(self.db_path))

        # Mode WAL (Write-Ahead Logging) : meilleur pour les écritures
        # concurrentes et plus performant que le mode rollback par défaut.
        # C'est le mode utilisé en production sur tous les systèmes embarqués.
        await self._connection.execute("PRAGMA journal_mode=WAL")

        # Synchronisation NORMAL : compromis bon entre durabilité et perf.
        # FULL serait plus sûr mais 2-3x plus lent.
        await self._connection.execute("PRAGMA synchronous=NORMAL")

        # Création du schéma (idempotent).
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()

        # Démarrage d'une nouvelle session.
        cursor = await self._connection.execute(
            "INSERT INTO sessions (started_at, source, notes) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), source, notes),
        )
        self._session_id = cursor.lastrowid
        await self._connection.commit()

        logger.info(
            "Session SQLite #%d démarrée (source: %s)", self._session_id, source
        )

    async def stop(self) -> None:
        """
        Clôt proprement la session courante et ferme la connexion.

        Cette méthode est idempotente : si elle est appelée plusieurs fois,
        seul le premier appel a un effet réel. Pratique en cas d'arrêt
        désordonné du système.
        """
        if self._connection is None:
            return

        if self._session_id is not None:
            try:
                await self._connection.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), self._session_id),
                )
                await self._connection.commit()
                logger.info(
                    "Session SQLite #%d clôturée — %d trames, %d détections, %d erreurs",
                    self._session_id,
                    self._frames_logged,
                    self._detections_logged,
                    self._write_errors,
                )
            except Exception as exc:
                logger.error("Impossible de clôturer la session : %s", exc)

        await self._connection.close()
        self._connection = None
        self._session_id = None

    # ─── Subscriber du bus ──────────────────────────────────────────

    async def on_frame(self, frame: RadarFrame) -> None:
        """
        Handler abonné à 'radar.frame' : insère la trame et ses détections.

        Pour chaque trame, on insère :
            - 1 ligne dans `frames` (avec target_count même si 0)
            - N lignes dans `detections` (1 par cible présente)

        Les deux insertions sont faites dans une SEULE transaction pour
        garantir l'atomicité : soit tout est écrit, soit rien.

        Si l'écriture échoue (disque plein, base corrompue...), on log
        et on incrémente le compteur d'erreurs, mais on ne lève pas :
        le pipeline temps réel doit continuer coûte que coûte.
        """
        if self._connection is None or self._session_id is None:
            logger.warning("DetectionLogger non démarré, frame ignorée")
            return

        try:
            # Insertion de la trame, récupération de son ID auto-généré.
            cursor = await self._connection.execute(
                """
                INSERT INTO frames (session_id, timestamp, target_count, raw_hex)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._session_id,
                    frame.timestamp.isoformat(),
                    frame.target_count,
                    frame.raw_hex,
                ),
            )
            frame_id = cursor.lastrowid

            # Insertion des détections (une par cible) en batch.
            if frame.targets:
                detection_rows = [
                    (
                        frame_id,
                        t.slot,
                        t.x_mm,
                        t.y_mm,
                        t.speed_cms,
                        t.distance_m,
                        t.angle_degrees,
                        t.state.value,
                        t.resolution_mm,
                    )
                    for t in frame.targets
                ]
                await self._connection.executemany(
                    """
                    INSERT INTO detections
                        (frame_id, slot, x_mm, y_mm, speed_cms,
                         distance_m, angle_deg, state, resolution_mm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    detection_rows,
                )

            await self._connection.commit()

            self._frames_logged += 1
            self._detections_logged += frame.target_count

        except Exception as exc:
            self._write_errors += 1
            logger.error("Erreur d'écriture SQLite : %s", exc)
            # Tentative de rollback pour ne pas laisser une transaction ouverte.
            try:
                await self._connection.rollback()
            except Exception:
                pass

    # ─── Introspection ──────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "session_id": self._session_id,
            "frames_logged": self._frames_logged,
            "detections_logged": self._detections_logged,
            "write_errors": self._write_errors,
        }