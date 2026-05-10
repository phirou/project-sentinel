"""
EventBus asynchrone : système de publication/abonnement interne à Sentinel.

Ce module fournit un bus d'événements typé permettant aux différents
composants de Sentinel (drivers, tracking, logger, WebSocket, AI) de
communiquer sans dépendances directes les uns aux autres.

Pattern utilisé : Observer (Pub/Sub) avec dispatch asynchrone.

Avantages :
- Découplage fort : les producteurs d'événements ne connaissent pas
  les consommateurs, et vice-versa.
- Testabilité : chaque composant se teste avec un EventBus mock.
- Extensibilité : ajouter un nouvel abonné ne touche pas le producteur.
- Concurrence propre : les handlers asynchrones tournent en parallèle
  sur la même boucle asyncio sans threads.

Usage typique :
    bus = EventBus()

    async def on_radar_frame(frame: RadarFrame) -> None:
        print(f'Cibles : {frame.target_count}')

    bus.subscribe('radar.frame', on_radar_frame)
    await bus.publish('radar.frame', frame)

Auteur : Léo
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Type alias pour un handler d'événement : une coroutine qui prend le
# payload de l'événement et ne retourne rien d'utile (ses effets sont
# par side-effect : log, persist, push WebSocket, etc.).
EventHandler = Callable[[Any], Awaitable[None]]


# ─────────────────────────────────────────────────────────────────────────────
# CONVENTIONS DE NOMMAGE DES ÉVÉNEMENTS
# ─────────────────────────────────────────────────────────────────────────────
# Les noms d'événements suivent une convention hiérarchique pointée :
#   <domaine>.<sous-domaine>.<action>
#
# Exemples :
#   - radar.frame              : trame radar fraîchement parsée
#   - radar.target.entered     : nouvelle cible apparue
#   - radar.target.lost        : cible disparue
#   - zone.alert.triggered     : alerte de zone déclenchée
#   - camera.snapshot.captured : photo capturée
#   - ai.classification.done   : classification IA terminée
#
# Cette convention permettra plus tard d'implémenter des wildcards
# (genre `radar.*` pour s'abonner à tout ce qui concerne le radar).
# ─────────────────────────────────────────────────────────────────────────────


class EventBus:
    """
    Bus d'événements asynchrone pour la communication inter-modules.

    Le bus maintient un registre d'abonnés par nom d'événement. Quand un
    événement est publié, tous les handlers abonnés sont appelés en parallèle
    via asyncio.gather().

    Cette classe est thread-safe au sens asyncio : toutes les opérations
    s'exécutent sur la boucle d'événements unique. Pour un usage multi-thread,
    il faudrait passer par run_coroutine_threadsafe().
    """

    def __init__(self) -> None:
        # Dictionnaire {nom_événement: liste de handlers}.
        # defaultdict évite d'avoir à vérifier l'existence de la clé.
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

        # Compteurs de statistiques pour debug et monitoring.
        self._publish_count: int = 0
        self._handler_errors: int = 0

    # ─── API d'abonnement ────────────────────────────────────────────

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """
        Abonne un handler asynchrone à un événement.

        Le même handler peut être abonné plusieurs fois au même événement
        (il sera alors appelé plusieurs fois). Le même handler peut aussi
        être abonné à plusieurs événements différents.

        Args:
            event_name: nom de l'événement, par convention en notation
                        pointée (ex: 'radar.frame').
            handler: coroutine qui sera appelée avec le payload de
                     l'événement à chaque publication.
        """
        self._handlers[event_name].append(handler)
        logger.debug(
            "Handler %s abonné à l'événement '%s' (total: %d)",
            handler.__name__,
            event_name,
            len(self._handlers[event_name]),
        )

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """
        Désabonne un handler d'un événement.

        Si le handler n'est pas abonné, l'appel est silencieux (pas d'erreur).
        Cela évite des erreurs lors de cleanup en série.
        """
        if handler in self._handlers.get(event_name, []):
            self._handlers[event_name].remove(handler)
            logger.debug(
                "Handler %s désabonné de l'événement '%s'",
                handler.__name__,
                event_name,
            )

    # ─── API de publication ──────────────────────────────────────────

    async def publish(self, event_name: str, payload: Any = None) -> None:
        """
        Publie un événement vers tous ses abonnés, en parallèle.

        Tous les handlers abonnés à event_name sont exécutés concurremment
        via asyncio.gather. Si un handler lève une exception, elle est
        capturée et loguée, mais n'interrompt pas les autres handlers
        ni la suite du programme.

        Args:
            event_name: nom de l'événement à publier.
            payload: donnée associée à l'événement (souvent une dataclass
                     ou un modèle Pydantic). Peut être None.
        """
        self._publish_count += 1

        handlers = self._handlers.get(event_name, [])
        if not handlers:
            # Pas d'abonné : pas grave, on log en debug seulement.
            logger.debug("Aucun handler pour l'événement '%s'", event_name)
            return

        # Lancement en parallèle de tous les handlers via gather.
        # return_exceptions=True permet de capturer les erreurs de chaque
        # handler individuellement plutôt que d'avorter le gather complet.
        results = await asyncio.gather(
            *(handler(payload) for handler in handlers),
            return_exceptions=True,
        )

        # Log des erreurs éventuelles. On ne lève pas, le bus doit rester
        # robuste : un handler qui plante ne doit pas casser tout Sentinel.
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                self._handler_errors += 1
                logger.error(
                    "Handler '%s' a levé une exception sur '%s' : %s",
                    handler.__name__,
                    event_name,
                    result,
                    exc_info=result,
                )

    # ─── Introspection / monitoring ─────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        """
        Retourne des statistiques d'usage du bus, utiles pour le monitoring.
        """
        return {
            "publish_count": self._publish_count,
            "handler_errors": self._handler_errors,
            "subscribed_events": len(self._handlers),
            "total_handlers": sum(len(hs) for hs in self._handlers.values()),
        }

    def __repr__(self) -> str:
        return (
            f"EventBus(events={len(self._handlers)}, "
            f"published={self._publish_count}, errors={self._handler_errors})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE GLOBALE (PATTERN SINGLETON LIGHT)
# ─────────────────────────────────────────────────────────────────────────────

# Un EventBus unique partagé par tout Sentinel.
# Note : pour les tests unitaires, il vaut mieux instancier un EventBus
# local plutôt que d'utiliser ce singleton (isolation des tests).
bus = EventBus()