"""
Application FastAPI : serveur web et WebSocket de Sentinel.

Ce module expose deux interfaces au monde extérieur :
    - WebSocket /ws/realtime : flux temps réel des détections radar
    - REST /api/status      : santé du système (frames count, uptime, etc.)
    - GET /                  : sert le dashboard HTML statique

L'application FastAPI est construite via une factory create_app() qui
prend en paramètre l'EventBus de l'application. Cette inversion de
dépendance permet d'instancier l'app avec un bus mock pour les tests.

Auteur : Léo
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sentinel.core.event_bus import EventBus
from sentinel.core.models import RadarFrame

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Chemin vers le dossier `web/` qui contient l'index.html et les assets.
# On utilise Path.resolve() pour obtenir un chemin absolu, robuste au
# répertoire de travail courant.
WEB_DIR: Path = (Path(__file__).parent.parent.parent / "web").resolve()


# ─────────────────────────────────────────────────────────────────────────────
# GESTIONNAIRE DE CONNEXIONS WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

class WebSocketBroadcaster:
    """
    Diffuse les événements radar à tous les clients WebSocket connectés.

    S'abonne à l'événement 'radar.frame' du bus et sérialise chaque trame
    en JSON pour la pousser à tous les navigateurs connectés simultanément.

    Si un client se déconnecte (onglet fermé, perte réseau), il est
    silencieusement retiré de la liste sans interrompre les autres.
    """

    def __init__(self) -> None:
        # Set plutôt que list : O(1) pour ajout/suppression et pas de doublons.
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepte une nouvelle connexion WebSocket et l'enregistre."""
        await websocket.accept()
        self._connections.add(websocket)
        logger.info(
            "Client WebSocket connecté (total: %d)", len(self._connections)
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Retire une connexion (typiquement quand le client part)."""
        self._connections.discard(websocket)
        logger.info(
            "Client WebSocket déconnecté (total: %d)", len(self._connections)
        )

    async def on_frame(self, frame: RadarFrame) -> None:
        """
        Handler appelé par l'EventBus à chaque trame radar.

        Sérialise la trame en JSON et la diffuse à tous les clients
        connectés. Les clients dont l'envoi échoue sont retirés du set.
        """
        if not self._connections:
            return  # personne ne nous écoute, on économise la sérialisation

        # Sérialisation JSON via Pydantic — gratuit grâce à nos modèles.
        # mode="json" garantit une sortie compatible JSON pur (datetime → str).
        payload = frame.model_dump(mode="json")
        message = {"type": "radar.frame", "data": payload}

        # On itère sur une COPIE du set parce qu'on peut modifier _connections
        # pendant le parcours (en cas de déconnexion détectée).
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                # Client parti pendant l'envoi : on le retire silencieusement.
                self._connections.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY DE L'APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def create_app(bus: EventBus) -> FastAPI:
    """
    Construit et retourne une instance configurée de l'application FastAPI.

    Args:
        bus: l'EventBus de Sentinel auquel l'app va s'abonner.

    Returns:
        Une instance FastAPI prête à être servie par uvicorn.
    """
    app = FastAPI(
        title="Sentinel Tactical API",
        description="Real-time radar surveillance system",
        version="0.1.0",
    )

    # Instanciation du broadcaster et abonnement au bus.
    broadcaster = WebSocketBroadcaster()
    bus.subscribe("radar.frame", broadcaster.on_frame)

    # Mémorisation de l'heure de démarrage pour calculer l'uptime.
    started_at = datetime.now(timezone.utc)

    # ─── Endpoint REST de status ─────────────────────────────────────

    @app.get("/api/status")
    async def get_status() -> dict:
        """Retourne l'état de santé général du système."""
        now = datetime.now(timezone.utc)
        uptime_seconds = (now - started_at).total_seconds()
        return {
            "status": "running",
            "started_at": started_at.isoformat(),
            "uptime_seconds": uptime_seconds,
            "websocket_clients": broadcaster.client_count,
            "event_bus": bus.stats,
        }

    # ─── Endpoint WebSocket temps réel ───────────────────────────────

    @app.websocket("/ws/realtime")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """
        Connexion WebSocket persistante pour le flux temps réel des détections.

        Le serveur pousse chaque RadarFrame en JSON. Le client peut envoyer
        des messages au serveur (par exemple ping/pong) qu'on consomme pour
        détecter une déconnexion propre.
        """
        await broadcaster.connect(websocket)
        try:
            # Boucle de réception : on lit ce que le client envoie.
            # Pour Phase 1, on n'attend rien de spécifique, mais ça maintient
            # la connexion ouverte et détecte les déconnexions.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.disconnect(websocket)

    # ─── Servir le dashboard HTML statique ───────────────────────────

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Sert le dashboard principal."""
        return FileResponse(WEB_DIR / "index.html")

    # Sert tout le contenu de web/ (CSS, JS) sous /static/
    if WEB_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=WEB_DIR),
            name="static",
        )

    return app