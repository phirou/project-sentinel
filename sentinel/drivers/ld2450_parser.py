"""
Parser binaire pour les trames du capteur radar HLK-LD2450.

Le LD2450 émet sur son port UART, à 256 000 bauds, des trames de 30 octets
toutes les 100 ms environ. Ce module est responsable de la conversion de
ces trames brutes (bytes) en objets Python typés (RadarFrame, Target).

Format de la trame (30 octets total) :
    [HEADER (4)] [CIBLE_1 (8)] [CIBLE_2 (8)] [CIBLE_3 (8)] [FOOTER (2)]

    Header  : AA FF 03 00     (signature de début)
    Footer  : 55 CC            (signature de fin)

Format d'un bloc cible (8 octets) :
    [X (2)] [Y (2)] [Speed (2)] [Resolution (2)]

    Toutes les valeurs sont en little-endian.
    X, Y et Speed utilisent la convention de signe non-standard de Hi-Link
    (le bit de poids fort code le signe à l'envers de la convention usuelle).

Référence : Hi-Link HLK-LD2450 Serial Communication Protocol Specification.

Auteur : Léo
"""

from __future__ import annotations

import logging
from typing import Optional

from sentinel.core.models import MAX_TARGETS_PER_FRAME, RadarFrame, Target

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DU PROTOCOLE
# ─────────────────────────────────────────────────────────────────────────────

# Signatures de début et de fin de trame, telles que documentées par Hi-Link.
FRAME_HEADER: bytes = bytes([0xAA, 0xFF, 0x03, 0x00])
FRAME_FOOTER: bytes = bytes([0x55, 0xCC])

# Tailles fixes (en octets) des différents segments.
FRAME_SIZE: int = 30
HEADER_SIZE: int = 4
FOOTER_SIZE: int = 2
TARGET_BLOCK_SIZE: int = 8

# Offsets de début et de fin de la zone des cibles dans la trame.
# La zone fait 24 octets = 3 cibles × 8 octets.
TARGETS_OFFSET: int = HEADER_SIZE                # 4
TARGETS_END: int = FRAME_SIZE - FOOTER_SIZE      # 28


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class LD2450ParseError(Exception):
    """Levée quand une trame LD2450 ne peut pas être parsée correctement."""


# ─────────────────────────────────────────────────────────────────────────────
# DÉCODAGE BAS NIVEAU
# ─────────────────────────────────────────────────────────────────────────────

def _decode_signed_16bit(data: bytes) -> int:
    """
    Décode 2 octets little-endian en entier signé, convention Hi-Link.

    Le LD2450 utilise une convention "magnitude + bit de signe" plutôt
    que le complément à deux standard :
        - Bit de poids fort (bit 15) à 1 → valeur NÉGATIVE
        - Bit de poids fort (bit 15) à 0 → valeur POSITIVE
        - Les 15 bits restants codent la magnitude (valeur absolue)

    C'est différent du complément à deux qu'on rencontre habituellement
    en informatique, mais c'est la convention documentée par Hi-Link.

    Args:
        data: exactement 2 octets (sinon ValueError).

    Returns:
        Valeur entière signée (entre -32767 et +32767).
    """
    if len(data) != 2:
        raise ValueError(f"_decode_signed_16bit attend 2 octets, reçu {len(data)}")

    # On lit la valeur brute non-signée en little-endian.
    raw = int.from_bytes(data, byteorder="little", signed=False)

    # Le bit de poids fort (bit 15) est le bit de signe selon Hi-Link.
    # 0x8000 = 1000 0000 0000 0000 en binaire.
    sign_bit = raw & 0x8000

    # Les 15 bits restants codent la magnitude (valeur absolue).
    # 0x7FFF = 0111 1111 1111 1111 en binaire.
    magnitude = raw & 0x7FFF

    # Convention Hi-Link : bit de signe à 1 = valeur négative,
    # bit de signe à 0 = valeur positive (convention standard).
    return -magnitude if sign_bit else magnitude


def _decode_unsigned_16bit(data: bytes) -> int:
    """
    Décode 2 octets little-endian en entier non signé.

    Utilisé pour la résolution, qui est toujours positive.

    Args:
        data: exactement 2 octets.

    Returns:
        Valeur entière non signée (entre 0 et 65535).
    """
    if len(data) != 2:
        raise ValueError(f"_decode_unsigned_16bit attend 2 octets, reçu {len(data)}")
    return int.from_bytes(data, byteorder="little", signed=False)


# ─────────────────────────────────────────────────────────────────────────────
# DÉCODAGE D'UN BLOC CIBLE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_target_block(data: bytes, slot: int) -> Optional[Target]:
    """
    Décode un bloc de 8 octets en un objet Target, ou None si le slot est vide.

    Le LD2450 remplit les slots cibles non utilisés par des zéros. Si tous
    les champs (X, Y, Speed) sont nuls, on considère que le slot ne contient
    pas de cible réelle.

    Args:
        data: exactement 8 octets correspondant à un bloc cible.
        slot: index du slot dans la trame (0, 1 ou 2), reporté dans le Target.

    Returns:
        Un objet Target si le slot contient une cible, None si vide.
    """
    if len(data) != TARGET_BLOCK_SIZE:
        raise ValueError(
            f"_parse_target_block attend {TARGET_BLOCK_SIZE} octets, reçu {len(data)}"
        )

    # Extraction des 4 champs selon les offsets du protocole.
    x_mm = _decode_signed_16bit(data[0:2])
    y_mm = _decode_signed_16bit(data[2:4])
    speed_cms = _decode_signed_16bit(data[4:6])
    resolution_mm = _decode_unsigned_16bit(data[6:8])

    # Détection d'un slot vide : Hi-Link met tout à zéro pour les cibles
    # inexistantes. La résolution peut être non nulle pour une vraie cible
    # immobile, donc on ne se base pas dessus pour la détection de vide.
    if x_mm == 0 and y_mm == 0 and speed_cms == 0:
        return None

    return Target(
        slot=slot,
        x_mm=x_mm,
        y_mm=y_mm,
        speed_cms=speed_cms,
        resolution_mm=resolution_mm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DÉCODAGE D'UNE TRAME COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

def parse_frame(data: bytes, include_raw: bool = False) -> RadarFrame:
    """
    Décode une trame complète de 30 octets en un objet RadarFrame.

    Cette fonction est le point d'entrée principal du parser. Elle valide
    la structure de la trame (taille, header, footer), puis extrait les
    cibles présentes.

    Args:
        data: trame brute de 30 octets reçue depuis l'UART.
        include_raw: si True, l'attribut raw_hex de la trame retournée
                     contiendra la représentation hexadécimale de data.
                     Utile en debug, à désactiver en production pour la perf.

    Returns:
        Un RadarFrame contenant les cibles détectées (0 à 3).

    Raises:
        LD2450ParseError: si la trame est malformée (mauvaise taille,
                          header ou footer invalide).
    """
    # --- 1. Validation de la taille de la trame ---
    if len(data) != FRAME_SIZE:
        raise LD2450ParseError(
            f"Taille de trame invalide : attendu {FRAME_SIZE} octets, reçu {len(data)}"
        )

    # --- 2. Validation du header de début ---
    if data[:HEADER_SIZE] != FRAME_HEADER:
        raise LD2450ParseError(
            f"Header invalide : attendu {FRAME_HEADER.hex(' ')}, "
            f"reçu {data[:HEADER_SIZE].hex(' ')}"
        )

    # --- 3. Validation du footer de fin ---
    if data[TARGETS_END:] != FRAME_FOOTER:
        raise LD2450ParseError(
            f"Footer invalide : attendu {FRAME_FOOTER.hex(' ')}, "
            f"reçu {data[TARGETS_END:].hex(' ')}"
        )

    # --- 4. Extraction des cibles ---
    # On parcourt les 3 slots possibles. Chaque slot fait TARGET_BLOCK_SIZE
    # octets, à partir de TARGETS_OFFSET.
    targets: list[Target] = []
    for slot in range(MAX_TARGETS_PER_FRAME):
        # Calcul des offsets pour ce slot dans la trame complète.
        block_start = TARGETS_OFFSET + slot * TARGET_BLOCK_SIZE
        block_end = block_start + TARGET_BLOCK_SIZE

        # Décodage du bloc. Renvoie None si le slot est vide.
        target = _parse_target_block(data[block_start:block_end], slot=slot)

        if target is not None:
            targets.append(target)

    # --- 5. Construction de la trame finale ---
    return RadarFrame(
        targets=targets,
        raw_hex=data.hex(" ") if include_raw else None,
    )