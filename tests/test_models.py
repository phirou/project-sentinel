"""
Tests unitaires pour les modèles de domaine (Target, RadarFrame).

Couvre :
    - Validation Pydantic (contraintes, types)
    - Computed fields (distance, angle, état cinématique)
    - Sérialisation JSON
"""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from sentinel.core.models import RadarFrame, Target, TargetState


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DU MODÈLE Target
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetModel:
    """Tests de la classe Target et de ses computed fields."""

    def test_create_valid_target(self) -> None:
        """Création d'une cible avec des données valides."""
        t = Target(slot=0, x_mm=1000, y_mm=2000, speed_cms=50, resolution_mm=100)
        assert t.slot == 0
        assert t.x_mm == 1000
        assert t.y_mm == 2000

    @pytest.mark.parametrize("invalid_slot", [-1, 3, 4, 100])
    def test_slot_must_be_0_to_2(self, invalid_slot: int) -> None:
        """slot hors de [0, 2] doit lever une ValidationError."""
        with pytest.raises(ValidationError):
            Target(
                slot=invalid_slot,
                x_mm=1000,
                y_mm=2000,
                speed_cms=50,
                resolution_mm=100,
            )

    def test_negative_resolution_rejected(self) -> None:
        """La résolution doit être positive ou nulle."""
        with pytest.raises(ValidationError):
            Target(
                slot=0, x_mm=1000, y_mm=2000,
                speed_cms=50, resolution_mm=-10,
            )


class TestTargetComputedFields:
    """Tests des propriétés calculées (distance, angle, état)."""

    def test_distance_pythagorean(self) -> None:
        """distance_mm = sqrt(x² + y²)."""
        t = Target(slot=0, x_mm=3000, y_mm=4000, speed_cms=0, resolution_mm=100)
        # 3-4-5 triangle : distance = 5000 mm
        assert t.distance_mm == pytest.approx(5000, abs=0.1)
        assert t.distance_m == pytest.approx(5.0, abs=0.001)

    def test_angle_straight_ahead(self) -> None:
        """Une cible droit devant (x=0) doit donner angle = 0°."""
        t = Target(slot=0, x_mm=0, y_mm=5000, speed_cms=0, resolution_mm=100)
        assert t.angle_degrees == pytest.approx(0.0, abs=0.1)

    def test_angle_positive_for_right(self) -> None:
        """Une cible à droite (x>0) doit donner un angle positif."""
        t = Target(slot=0, x_mm=1000, y_mm=1000, speed_cms=0, resolution_mm=100)
        # 45° à droite
        assert t.angle_degrees == pytest.approx(45.0, abs=0.1)

    def test_angle_negative_for_left(self) -> None:
        """Une cible à gauche (x<0) doit donner un angle négatif."""
        t = Target(slot=0, x_mm=-1000, y_mm=1000, speed_cms=0, resolution_mm=100)
        assert t.angle_degrees == pytest.approx(-45.0, abs=0.1)

    @pytest.mark.parametrize("speed_cms,expected_state", [
        (0, TargetState.STATIONARY),
        (3, TargetState.STATIONARY),     # bruit Doppler, dans le seuil
        (-4, TargetState.STATIONARY),    # idem
        (50, TargetState.RECEDING),      # s'éloigne
        (-50, TargetState.APPROACHING),  # s'approche
    ])
    def test_kinematic_state(self, speed_cms: int, expected_state: TargetState) -> None:
        """Vérifie la dérivation de l'état cinématique selon la vitesse."""
        t = Target(slot=0, x_mm=1000, y_mm=1000, speed_cms=speed_cms, resolution_mm=100)
        assert t.state == expected_state


class TestTargetSerialization:
    """Tests de la sérialisation JSON (utilisée par le WebSocket et l'API)."""

    def test_to_dict_includes_computed_fields(self) -> None:
        """Le dump dict doit inclure les computed fields."""
        t = Target(slot=0, x_mm=3000, y_mm=4000, speed_cms=50, resolution_mm=100)
        d = t.model_dump()
        assert "distance_m" in d
        assert "angle_degrees" in d
        assert "state" in d

    def test_to_json_is_valid_json(self) -> None:
        """Le JSON dump doit être parsable par json.loads."""
        t = Target(slot=0, x_mm=1000, y_mm=2000, speed_cms=50, resolution_mm=100)
        json_str = t.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["slot"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DU MODÈLE RadarFrame
# ─────────────────────────────────────────────────────────────────────────────

class TestRadarFrameModel:
    """Tests de la trame radar (conteneur de cibles)."""

    def test_empty_frame(self) -> None:
        """Une trame sans cible doit avoir target_count = 0."""
        f = RadarFrame()
        assert f.target_count == 0
        assert f.has_targets is False

    def test_frame_with_targets(self) -> None:
        """target_count doit refléter le nombre de cibles."""
        targets = [
            Target(slot=0, x_mm=1000, y_mm=2000, speed_cms=0, resolution_mm=100),
            Target(slot=1, x_mm=2000, y_mm=3000, speed_cms=0, resolution_mm=100),
        ]
        f = RadarFrame(targets=targets)
        assert f.target_count == 2
        assert f.has_targets is True