"""
Tests unitaires pour l'EventBus.

Couvre :
    - Abonnement / désabonnement
    - Dispatch d'événements à plusieurs handlers
    - Robustesse aux exceptions des handlers
    - Statistiques
"""

from __future__ import annotations

import pytest

from sentinel.core.event_bus import EventBus


class TestEventBusBasic:
    """Tests des opérations de base : subscribe, publish."""

    async def test_publish_without_subscribers(self) -> None:
        """Publier sans abonné ne doit pas planter."""
        bus = EventBus()
        await bus.publish("test.event", {"data": 42})
        assert bus.stats["publish_count"] == 1

    async def test_handler_receives_payload(self) -> None:
        """Un handler abonné doit recevoir le payload publié."""
        bus = EventBus()
        received = []

        async def handler(payload):
            received.append(payload)

        bus.subscribe("test.event", handler)
        await bus.publish("test.event", "hello")

        assert received == ["hello"]

    async def test_multiple_handlers_all_called(self) -> None:
        """Plusieurs handlers abonnés au même événement reçoivent tous."""
        bus = EventBus()
        calls = []

        async def handler_a(payload):
            calls.append(("a", payload))

        async def handler_b(payload):
            calls.append(("b", payload))

        bus.subscribe("event", handler_a)
        bus.subscribe("event", handler_b)
        await bus.publish("event", 42)

        assert ("a", 42) in calls
        assert ("b", 42) in calls
        assert len(calls) == 2


class TestEventBusErrorHandling:
    """Tests de robustesse : un handler qui plante ne casse pas le bus."""

    async def test_failing_handler_does_not_break_others(self) -> None:
        """Si un handler lève, les autres handlers doivent quand même être appelés."""
        bus = EventBus()
        good_received = []

        async def failing_handler(payload):
            raise RuntimeError("boom")

        async def good_handler(payload):
            good_received.append(payload)

        bus.subscribe("event", failing_handler)
        bus.subscribe("event", good_handler)

        await bus.publish("event", "data")

        # Le bon handler doit avoir reçu malgré l'erreur de l'autre.
        assert good_received == ["data"]
        # Les stats doivent refléter l'erreur.
        assert bus.stats["handler_errors"] == 1


class TestEventBusUnsubscribe:
    """Tests de désabonnement."""

    async def test_unsubscribe_stops_delivery(self) -> None:
        """Après unsubscribe, le handler ne reçoit plus rien."""
        bus = EventBus()
        received = []

        async def handler(payload):
            received.append(payload)

        bus.subscribe("event", handler)
        await bus.publish("event", 1)
        assert len(received) == 1

        bus.unsubscribe("event", handler)
        await bus.publish("event", 2)
        # Le handler ne doit pas avoir reçu le second event.
        assert len(received) == 1

    async def test_unsubscribe_unknown_handler_is_silent(self) -> None:
        """Désabonner un handler non abonné ne doit pas planter."""
        bus = EventBus()

        async def handler(payload):
            pass

        # Ce désabonnement doit être silencieux.
        bus.unsubscribe("event", handler)


class TestEventBusStats:
    """Tests des statistiques internes."""

    async def test_publish_count_increments(self) -> None:
        bus = EventBus()
        assert bus.stats["publish_count"] == 0

        await bus.publish("event")
        await bus.publish("event")
        await bus.publish("other")

        assert bus.stats["publish_count"] == 3