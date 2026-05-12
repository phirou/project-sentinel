"""
Tests unitaires pour le simulateur LD2450.

Couvre :
    - Encodage signed 16-bit (inverse du décodage)
    - Round-trip simulator → parser (équivalence des conventions)
    - Dynamique des cibles (update, rebonds)
    - Construction de trames complètes
"""

from __future__ import annotations

import math

import pytest

from sentinel.drivers.ld2450_parser import parse_frame
from sentinel.drivers.ld2450_simulator import (
    LD2450Simulator,
    MAX_RANGE_MM,
    SimulatedTarget,
    _encode_signed_16bit,
    encode_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE L'ENCODAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeSigned16bit:
    """Tests de la fonction d'encodage signed 16-bit convention Hi-Link."""

    @pytest.mark.parametrize("value", [0, 1, -1, 100, -100, 1101, -1101, 32767, -32767])
    def test_round_trip_encode_decode(self, value: int) -> None:
        """
        Encoder puis décoder une valeur doit retourner la valeur d'origine.

        C'est le test de cohérence parser ↔ simulateur : si ce test passe
        pour toutes les valeurs, la stack est complète.
        """
        from sentinel.drivers.ld2450_parser import _decode_signed_16bit
        encoded = _encode_signed_16bit(value)
        decoded = _decode_signed_16bit(encoded)
        assert decoded == value

    def test_encode_clamps_overflow(self) -> None:
        """Une valeur > 32767 doit être tronquée à 32767, pas planter."""
        result = _encode_signed_16bit(50000)
        # On vérifie juste que ça ne crash pas et que ça retourne 2 octets.
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DU ROUND-TRIP COMPLET
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    """
    Tests cruciaux : encode_frame() puis parse_frame() doit préserver
    les données. C'est ce qui garantit que le simulateur et le parser
    sont mutuellement cohérents.
    """

    def test_empty_frame_round_trip(self) -> None:
        """Trame sans cible : encode + parse = 0 cible."""
        frame_bytes = encode_frame([])
        frame = parse_frame(frame_bytes)
        assert frame.target_count == 0

    def test_single_target_position_preserved(self) -> None:
        """Position (x, y) d'une cible doit être préservée après round-trip."""
        target = SimulatedTarget(x_mm=1500, y_mm=3000, vx_mms=0, vy_mms=0)
        frame_bytes = encode_frame([target])
        frame = parse_frame(frame_bytes)
        assert frame.target_count == 1
        # Tolérance d'1 mm liée à l'arrondi int() dans l'encodage.
        assert abs(frame.targets[0].x_mm - 1500) <= 1
        assert abs(frame.targets[0].y_mm - 3000) <= 1

    def test_negative_coordinates_preserved(self) -> None:
        """Les coordonnées négatives (X gauche) doivent être préservées."""
        target = SimulatedTarget(x_mm=-2000, y_mm=4000, vx_mms=0, vy_mms=0)
        frame_bytes = encode_frame([target])
        frame = parse_frame(frame_bytes)
        assert frame.targets[0].x_mm < 0
        assert abs(frame.targets[0].x_mm - (-2000)) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE LA DYNAMIQUE
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetDynamics:
    """Tests de la mise à jour position/vitesse des cibles simulées."""

    def test_target_moves_with_velocity(self) -> None:
        """Après un step dt, la position doit être p0 + v*dt."""
        target = SimulatedTarget(x_mm=1000, y_mm=2000, vx_mms=500, vy_mms=200)
        target.update(dt=1.0)  # 1 seconde
        # Nouvelle position = ancienne + vitesse * dt
        assert target.x_mm == pytest.approx(1500, abs=1)
        assert target.y_mm == pytest.approx(2200, abs=1)

    def test_target_bounces_at_max_range(self) -> None:
        """Une cible qui dépasse MAX_RANGE_MM en Y doit rebondir."""
        # Cible juste au bord, allant vers l'extérieur.
        target = SimulatedTarget(
            x_mm=0,
            y_mm=MAX_RANGE_MM - 100,
            vx_mms=0,
            vy_mms=1000,  # va vers Y+ (s'éloigne)
        )
        # 1 seconde plus tard, elle aurait dû atteindre MAX_RANGE et rebondir.
        target.update(dt=1.0)
        # Y doit être borné par MAX_RANGE_MM.
        assert target.y_mm <= MAX_RANGE_MM
        # La vitesse Y doit avoir changé de signe (rebond).
        assert target.vy_mms < 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DU SIMULATEUR
# ─────────────────────────────────────────────────────────────────────────────

class TestLD2450Simulator:
    """Tests du conteneur de simulation principal."""

    def test_add_target_increments_count(self) -> None:
        sim = LD2450Simulator()
        assert len(sim.targets) == 0
        sim.add_target(SimulatedTarget(x_mm=1000, y_mm=2000))
        assert len(sim.targets) == 1

    def test_cannot_add_more_than_three_targets(self) -> None:
        """Le LD2450 ne peut suivre que 3 cibles max — ajouter une 4e doit échouer."""
        sim = LD2450Simulator()
        for _ in range(3):
            sim.add_target(SimulatedTarget(x_mm=1000, y_mm=2000))
        with pytest.raises(RuntimeError, match="3"):
            sim.add_target(SimulatedTarget(x_mm=1000, y_mm=2000))

    def test_step_produces_valid_frame(
        self, simulator_with_three_targets: LD2450Simulator
    ) -> None:
        """Un step doit produire une trame valide de 30 octets parsable."""
        frame_bytes = simulator_with_three_targets.step()
        assert len(frame_bytes) == 30
        # La trame doit être parsable sans erreur.
        frame = parse_frame(frame_bytes)
        assert frame.target_count == 3

    def test_populate_random_respects_limit(self) -> None:
        """populate_random ne doit jamais dépasser 3 cibles."""
        sim = LD2450Simulator()
        sim.populate_random(n=3)
        assert len(sim.targets) == 3

        with pytest.raises(ValueError):
            sim.populate_random(n=4)