"""
Fixtures pytest partagées entre tous les tests Sentinel.

conftest.py est un fichier spécial reconnu automatiquement par pytest :
toutes les fixtures qui y sont définies sont disponibles dans tous les
fichiers de test du même dossier (et sous-dossiers).

Pattern utilisé : Arrange/Act/Assert. Les fixtures préparent l'état
("Arrange"), les tests effectuent une action ("Act") et vérifient
le résultat ("Assert").
"""

from __future__ import annotations

import pytest

from sentinel.drivers.ld2450_parser import (
    FRAME_FOOTER,
    FRAME_HEADER,
)
from sentinel.drivers.ld2450_simulator import (
    LD2450Simulator,
    SimulatedTarget,
    encode_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES DE TRAMES BINAIRES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_frame_bytes() -> bytes:
    """Trame valide sans aucune cible (les 3 slots vides)."""
    return FRAME_HEADER + bytes(24) + FRAME_FOOTER


@pytest.fixture
def one_target_frame_bytes() -> bytes:
    """
    Trame avec une cible unique à (X=1101, Y=4508, V=+50 cm/s).
    Slots 1 et 2 vides.
    """
    target = SimulatedTarget(
        x_mm=1101,
        y_mm=4508,
        # Pour que la vitesse radiale soit exactement +50 cm/s, on
        # construit un vecteur vitesse aligné avec le vecteur radial.
        vx_mms=(1101 / 4639.86) * 500,  # 500 mm/s radial
        vy_mms=(4508 / 4639.86) * 500,
    )
    return encode_frame([target])


@pytest.fixture
def three_target_frame_bytes() -> bytes:
    """Trame avec 3 cibles différentes (slots 0, 1, 2 tous remplis)."""
    targets = [
        SimulatedTarget(x_mm=1000, y_mm=2000, vx_mms=200, vy_mms=100),
        SimulatedTarget(x_mm=-1500, y_mm=3500, vx_mms=-100, vy_mms=-200),
        SimulatedTarget(x_mm=500, y_mm=5000, vx_mms=0, vy_mms=300),
    ]
    return encode_frame(targets)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES DE SIMULATEURS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simulator_with_one_target() -> LD2450Simulator:
    """Simulateur préchargé avec une cible déterministe."""
    sim = LD2450Simulator()
    sim.add_target(SimulatedTarget(x_mm=2000, y_mm=4000, vx_mms=500, vy_mms=0))
    return sim


@pytest.fixture
def simulator_with_three_targets() -> LD2450Simulator:
    """Simulateur préchargé avec 3 cibles (capacité maximale du LD2450)."""
    sim = LD2450Simulator()
    sim.add_target(SimulatedTarget(x_mm=1000, y_mm=2000, vx_mms=100, vy_mms=100))
    sim.add_target(SimulatedTarget(x_mm=-1000, y_mm=3000, vx_mms=-100, vy_mms=100))
    sim.add_target(SimulatedTarget(x_mm=0, y_mm=5000, vx_mms=0, vy_mms=-100))
    return sim