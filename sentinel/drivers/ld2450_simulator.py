"""
Simulateur du capteur radar HLK-LD2450.

Ce module génère des trames LD2450 binaires conformes au protocole réel,
permettant de développer et tester toute la stack Sentinel (parser, backend,
dashboard, fusion capteurs) sans dépendre du matériel physique.

Le simulateur modélise des cibles avec une dynamique simple (position +
vitesse linéaire) qui rebondissent sur les bords du champ de vue radar.
Les trames sont sérialisées au format binaire LD2450 et peuvent être
décodées par le même parser que celles du vrai capteur.

Usage typique :
    sim = LD2450Simulator()
    sim.add_target(SimulatedTarget(x_mm=1000, y_mm=3000, vx_mms=200, vy_mms=-100))
    async for frame_bytes in sim.stream():
        # frame_bytes est une trame de 30 octets, comme depuis un vrai UART
        ...

Auteur : Léo
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import AsyncIterator

from sentinel.core.models import (
    MAX_TARGETS_PER_FRAME,
    RADAR_FOV_DEGREES,
    RADAR_MAX_RANGE_METERS,
)
from sentinel.drivers.ld2450_parser import (
    FRAME_FOOTER,
    FRAME_HEADER,
    FRAME_SIZE,
    TARGET_BLOCK_SIZE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

# Le LD2450 émet une trame toutes les ~100 ms (10 Hz).
# C'est la cadence qu'on reproduit pour rester réaliste.
DEFAULT_FRAME_PERIOD_S: float = 0.1

# Bornes spatiales de la simulation, en millimètres.
# On contraint les cibles dans le champ de vue physique du capteur.
MAX_RANGE_MM: int = int(RADAR_MAX_RANGE_METERS * 1000)
HALF_FOV_RAD: float = math.radians(RADAR_FOV_DEGREES / 2.0)

# Limites de vitesse réalistes pour des cibles humaines/véhicules, en mm/s.
# 5 m/s ≈ 18 km/h (jogging soutenu, vélo lent).
MAX_VELOCITY_MMS: int = 5000


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE DE CIBLE SIMULÉE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedTarget:
    """
    Cible simulée avec position et vitesse en coordonnées cartésiennes.

    Contrairement à la classe Target du domaine (issue d'une mesure réelle),
    SimulatedTarget porte aussi les composantes vectorielles de vitesse
    (vx, vy) nécessaires pour faire évoluer la cible dans le temps.
    """

    x_mm: float
    y_mm: float
    vx_mms: float = 0.0  # vitesse selon X en mm/s (positif = vers la droite)
    vy_mms: float = 0.0  # vitesse selon Y en mm/s (positif = s'éloigne du radar)
    resolution_mm: int = 100  # résolution radar typique pour une cible humaine

    def update(self, dt: float) -> None:
        """
        Fait évoluer la cible d'un pas de temps dt (secondes).

        Intégration d'Euler simple : nouvelle position = position + vitesse × dt.
        Si la cible sort du champ de vue, elle "rebondit" en inversant la
        composante de vitesse correspondante.
        """
        # Mise à jour de la position par intégration de la vitesse.
        self.x_mm += self.vx_mms * dt
        self.y_mm += self.vy_mms * dt

        # Rebond en Y : si la cible sort par devant (trop loin) ou par derrière
        # (passe derrière le radar), on inverse vy. La cible reste dans la zone
        # [100 mm, MAX_RANGE_MM] devant le radar.
        if self.y_mm < 100:
            self.y_mm = 100
            self.vy_mms = abs(self.vy_mms)  # force vers l'avant
        elif self.y_mm > MAX_RANGE_MM:
            self.y_mm = MAX_RANGE_MM
            self.vy_mms = -abs(self.vy_mms)  # force vers le radar

        # Rebond en X : on contraint la cible dans le cône angulaire du FoV.
        # À une distance Y donnée, X max = Y × tan(HALF_FOV).
        x_limit = self.y_mm * math.tan(HALF_FOV_RAD)
        if self.x_mm > x_limit:
            self.x_mm = x_limit
            self.vx_mms = -abs(self.vx_mms)
        elif self.x_mm < -x_limit:
            self.x_mm = -x_limit
            self.vx_mms = abs(self.vx_mms)

    @property
    def radial_speed_cms(self) -> int:
        """
        Vitesse radiale (signée) projetée sur l'axe radar→cible, en cm/s.

        C'est ce que mesurerait un vrai radar par effet Doppler : seule la
        composante de vitesse alignée avec la ligne radar-cible est observable.
        Convention : positive si la cible s'éloigne, négative si elle s'approche.
        """
        distance = math.hypot(self.x_mm, self.y_mm)
        if distance < 1.0:
            return 0  # protection contre la division par zéro

        # Produit scalaire (vitesse · direction radiale) / norme(direction)
        radial_velocity_mms = (
            self.vx_mms * self.x_mm + self.vy_mms * self.y_mm
        ) / distance

        # Conversion mm/s → cm/s, arrondie à l'entier.
        return int(round(radial_velocity_mms / 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE BINAIRE (l'inverse du parser)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_signed_16bit(value: int) -> bytes:
    """
    Encode un entier signé en 2 octets little-endian, convention Hi-Link.

    Inverse de _decode_signed_16bit dans le parser. Convention :
        - Valeur négative → bit de poids fort à 1
        - Valeur positive → bit de poids fort à 0
        - Les 15 bits restants codent la magnitude.
    """
    # Borne pour rester dans la plage 15-bit (les magnitudes max).
    magnitude = min(abs(int(value)), 0x7FFF)

    if value < 0:
        # Bit de signe à 1 + magnitude.
        raw = 0x8000 | magnitude
    else:
        # Bit de signe à 0, juste la magnitude.
        raw = magnitude

    return raw.to_bytes(2, byteorder="little", signed=False)


def _encode_target_block(target: SimulatedTarget | None) -> bytes:
    """
    Encode une cible simulée en bloc de 8 octets, ou bloc vide si target=None.
    """
    if target is None:
        return bytes(TARGET_BLOCK_SIZE)  # 8 octets à zéro

    return (
        _encode_signed_16bit(int(target.x_mm))
        + _encode_signed_16bit(int(target.y_mm))
        + _encode_signed_16bit(target.radial_speed_cms)
        + target.resolution_mm.to_bytes(2, byteorder="little", signed=False)
    )


def encode_frame(targets: list[SimulatedTarget]) -> bytes:
    """
    Construit une trame binaire LD2450 complète (30 octets) à partir des cibles.

    Si on lui passe moins de 3 cibles, les slots restants sont remplis avec
    des blocs vides (tous à zéro), comme le ferait le vrai capteur.
    """
    if len(targets) > MAX_TARGETS_PER_FRAME:
        raise ValueError(
            f"Maximum {MAX_TARGETS_PER_FRAME} cibles par trame, reçu {len(targets)}"
        )

    # Construction de la zone des cibles : on encode chaque slot,
    # vide ou non, jusqu'à atteindre MAX_TARGETS_PER_FRAME.
    target_blocks = b""
    for slot in range(MAX_TARGETS_PER_FRAME):
        target = targets[slot] if slot < len(targets) else None
        target_blocks += _encode_target_block(target)

    # Assemblage final : header + cibles + footer.
    frame = FRAME_HEADER + target_blocks + FRAME_FOOTER
    assert len(frame) == FRAME_SIZE, f"Trame de taille incorrecte : {len(frame)}"

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LD2450Simulator:
    """
    Simulateur de capteur LD2450 émettant des trames binaires en continu.

    Le simulateur maintient une liste de cibles avec leur dynamique propre.
    À chaque tick (intervalle de DEFAULT_FRAME_PERIOD_S secondes), il met
    à jour les positions et émet une trame conforme au protocole LD2450.

    Le mode async permet d'intégrer le simulateur dans une stack asyncio
    sans bloquer la boucle d'événements.
    """

    targets: list[SimulatedTarget] = field(default_factory=list)
    frame_period_s: float = DEFAULT_FRAME_PERIOD_S

    def add_target(self, target: SimulatedTarget) -> None:
        """Ajoute une cible à la simulation (max 3)."""
        if len(self.targets) >= MAX_TARGETS_PER_FRAME:
            raise RuntimeError(
                f"Le LD2450 ne peut suivre que {MAX_TARGETS_PER_FRAME} cibles maximum"
            )
        self.targets.append(target)

    def populate_random(self, n: int = 2) -> None:
        """
        Peuple la simulation avec n cibles placées aléatoirement dans le FoV.

        Pratique pour démarrer rapidement une démo sans avoir à scénariser
        manuellement les positions et vitesses.
        """
        if n > MAX_TARGETS_PER_FRAME:
            raise ValueError(f"Maximum {MAX_TARGETS_PER_FRAME} cibles aléatoires")

        self.targets.clear()
        for _ in range(n):
            # Position aléatoire dans le cône FoV à distance moyenne.
            distance = random.uniform(1500, MAX_RANGE_MM * 0.8)
            angle = random.uniform(-HALF_FOV_RAD * 0.8, HALF_FOV_RAD * 0.8)
            x = distance * math.sin(angle)
            y = distance * math.cos(angle)

            # Vitesse aléatoire raisonnable (équivalent marche/course).
            vx = random.uniform(-1500, 1500)
            vy = random.uniform(-1500, 1500)

            self.targets.append(SimulatedTarget(x_mm=x, y_mm=y, vx_mms=vx, vy_mms=vy))

    def step(self) -> bytes:
        """
        Avance la simulation d'un pas de temps et retourne la trame résultante.

        Méthode synchrone, utile pour les tests unitaires où on veut contrôler
        précisément le déroulement du temps simulé.
        """
        for target in self.targets:
            target.update(self.frame_period_s)
        return encode_frame(self.targets)

    async def stream(self) -> AsyncIterator[bytes]:
        """
        Itérateur asynchrone qui émet une trame binaire à chaque période.

        Le sleep asyncio.sleep() libère la boucle d'événements entre les
        trames, ce qui permet à d'autres coroutines (parser, WebSocket,
        dashboard) de travailler en parallèle sur la même boucle.

        Usage :
            async for frame_bytes in simulator.stream():
                frame = parse_frame(frame_bytes)
                ...
        """
        logger.info(
            "Démarrage du simulateur LD2450 : %d cible(s), période %.0f ms",
            len(self.targets),
            self.frame_period_s * 1000,
        )

        while True:
            frame_bytes = self.step()
            yield frame_bytes
            await asyncio.sleep(self.frame_period_s)