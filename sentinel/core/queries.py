"""
Requêtes SQL haut niveau pour la base Sentinel.

Ce module expose des fonctions async qui interrogent la base SQLite et
retournent des données structurées prêtes à être sérialisées en JSON
(API REST) ou consommées par des scripts (CLI, replay, export).

Toutes les fonctions ouvrent leur propre connexion lecture seule via
aiosqlite, ce qui évite tout conflit avec le DetectionLogger qui écrit
en parallèle. SQLite gère parfaitement la concurrence lecture/écriture
en mode WAL (configuré dans persistence.py).

Pattern utilisé : Repository — la couche d'accès aux données est isolée
du reste de l'application, qui ne fait que consommer ces fonctions.

Auteur : Léo
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA CONNEXION
# ─────────────────────────────────────────────────────────────────────────────

# Chemin par défaut, surchargeable au cas où on aurait plusieurs bases
# (production, replay, tests...).
DEFAULT_DB_PATH: str = "data/sentinel.db"


async def _open_readonly(db_path: str) -> aiosqlite.Connection:
    """
    Ouvre la base SQLite en mode lecture seule.

    L'option `?mode=ro` garantit qu'on ne risque pas d'écrire par erreur
    depuis les queries. C'est une bonne pratique défensive : si un bug
    voulait écrire ici, on aurait une erreur claire au lieu d'une
    corruption silencieuse.
    """
    # uri=True permet de passer un URI SQLite avec options.
    conn = await aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True)

    # row_factory : on récupère les lignes comme des dicts (clés = noms
    # de colonnes), beaucoup plus lisible que les tuples positionnels.
    conn.row_factory = aiosqlite.Row

    return conn


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convertit une Row aiosqlite en dict standard pour la sérialisation JSON."""
    return {key: row[key] for key in row.keys()}


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTES SUR LES SESSIONS
# ─────────────────────────────────────────────────────────────────────────────

async def list_sessions(
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Liste les sessions enregistrées, les plus récentes en premier.

    Pour chaque session, on enrichit avec le nombre de frames et de
    détections qu'elle contient — info pratique pour l'UI.
    """
    conn = await _open_readonly(db_path)
    try:
        cursor = await conn.execute(
            """
            SELECT
                s.id,
                s.started_at,
                s.ended_at,
                s.source,
                s.notes,
                COUNT(DISTINCT f.id) AS frame_count,
                COUNT(d.id)          AS detection_count
            FROM sessions s
            LEFT JOIN frames f     ON f.session_id = s.id
            LEFT JOIN detections d ON d.frame_id = f.id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await conn.close()


async def get_session(
    session_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict[str, Any]]:
    """Retourne les détails d'une session par son ID, ou None si absente."""
    conn = await _open_readonly(db_path)
    try:
        cursor = await conn.execute(
            """
            SELECT
                s.*,
                COUNT(DISTINCT f.id) AS frame_count,
                COUNT(d.id)          AS detection_count
            FROM sessions s
            LEFT JOIN frames f     ON f.session_id = s.id
            LEFT JOIN detections d ON d.frame_id = f.id
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTES SUR LES DÉTECTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def list_detections(
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 100,
    since: Optional[datetime] = None,
    session_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Liste les détections, avec filtres optionnels.

    Args:
        limit:      nombre max de détections retournées (les plus récentes).
        since:      ne retourner que les détections postérieures à cette date.
        session_id: filtrer sur une session donnée.

    Retourne une liste de dicts triés par timestamp décroissant.
    Chaque détection inclut le timestamp de la frame associée (jointure).
    """
    # Construction dynamique de la requête en fonction des filtres fournis.
    # On utilise des paramètres SQL (placeholders ?) pour éviter toute
    # injection — JAMAIS de concaténation de strings SQL avec des inputs.
    where_clauses: list[str] = []
    params: list[Any] = []

    if since is not None:
        where_clauses.append("f.timestamp >= ?")
        params.append(since.isoformat())

    if session_id is not None:
        where_clauses.append("f.session_id = ?")
        params.append(session_id)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            d.id,
            d.frame_id,
            d.slot,
            d.x_mm,
            d.y_mm,
            d.speed_cms,
            d.distance_m,
            d.angle_deg,
            d.state,
            d.resolution_mm,
            f.timestamp AS frame_timestamp,
            f.session_id
        FROM detections d
        JOIN frames f ON d.frame_id = f.id
        {where_sql}
        ORDER BY f.timestamp DESC, d.slot ASC
        LIMIT ?
    """
    params.append(limit)

    conn = await _open_readonly(db_path)
    try:
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await conn.close()


async def get_detection_stats(
    db_path: str = DEFAULT_DB_PATH,
    session_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Retourne des statistiques agrégées sur les détections.

    Si session_id est fourni, les stats sont limitées à cette session.
    Sinon, elles couvrent toute la base.

    Inclut :
        - Total des frames et détections
        - Répartition par état cinématique (approaching/receding/stationary)
        - Distance moyenne et max
    """
    where_clauses: list[str] = []
    params: list[Any] = []

    if session_id is not None:
        where_clauses.append("f.session_id = ?")
        params.append(session_id)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = await _open_readonly(db_path)
    try:
        # Stats globales
        cursor = await conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT f.id)             AS frame_count,
                COUNT(d.id)                       AS detection_count,
                COALESCE(AVG(d.distance_m), 0.0)  AS avg_distance_m,
                COALESCE(MAX(d.distance_m), 0.0)  AS max_distance_m,
                COALESCE(MIN(d.distance_m), 0.0)  AS min_distance_m
            FROM frames f
            LEFT JOIN detections d ON d.frame_id = f.id
            {where_sql}
            """,
            params,
        )
        global_stats = _row_to_dict(await cursor.fetchone())

        # Répartition par état
        cursor = await conn.execute(
            f"""
            SELECT
                d.state,
                COUNT(*) AS count,
                ROUND(AVG(d.distance_m), 2) AS avg_distance_m
            FROM detections d
            JOIN frames f ON d.frame_id = f.id
            {where_sql}
            GROUP BY d.state
            """,
            params,
        )
        state_rows = await cursor.fetchall()
        by_state = {r["state"]: _row_to_dict(r) for r in state_rows}

        return {
            **global_stats,
            "by_state": by_state,
        }
    finally:
        await conn.close()