"""
Tests unitaires pour le parser LD2450.

Couvre :
    - Validation du format (taille, header, footer)
    - Décodage des valeurs signées (convention Hi-Link)
    - Détection des slots vides
    - Gestion des erreurs (trames malformées)
"""

from __future__ import annotations

import pytest

from sentinel.drivers.ld2450_parser import (
    FRAME_FOOTER,
    FRAME_HEADER,
    FRAME_SIZE,
    LD2450ParseError,
    _decode_signed_16bit,
    _decode_unsigned_16bit,
    parse_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DES PRIMITIVES DE DÉCODAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeSigned16bit:
    """Tests de la fonction de décodage signed 16-bit convention Hi-Link."""

    @pytest.mark.parametrize("data,expected", [
        # Valeurs positives : MSB = 0
        (b"\x00\x00", 0),
        (b"\x01\x00", 1),
        (b"\xff\x00", 255),
        (b"\x00\x01", 256),
        (b"\x4d\x04", 1101),     # exemple de la doc Hi-Link
        (b"\xff\x7f", 32767),    # max positif
        # Valeurs négatives : MSB = 1
        (b"\x01\x80", -1),
        (b"\x4d\x84", -1101),
        (b"\xff\xff", -32767),   # max négatif
    ])
    def test_decode_known_values(self, data: bytes, expected: int) -> None:
        """Vérifie le décodage sur des valeurs de référence."""
        assert _decode_signed_16bit(data) == expected

    def test_raises_on_wrong_length(self) -> None:
        """Doit lever une ValueError si on passe autre chose que 2 octets."""
        with pytest.raises(ValueError):
            _decode_signed_16bit(b"\x00")
        with pytest.raises(ValueError):
            _decode_signed_16bit(b"\x00\x00\x00")


class TestDecodeUnsigned16bit:
    """Tests du décodage unsigned 16-bit standard (résolution)."""

    @pytest.mark.parametrize("data,expected", [
        (b"\x00\x00", 0),
        (b"\x80\x00", 128),
        (b"\xff\xff", 65535),  # max unsigned 16-bit
    ])
    def test_decode_known_values(self, data: bytes, expected: int) -> None:
        assert _decode_unsigned_16bit(data) == expected


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DU PARSER DE TRAME COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFrame:
    """Tests de parse_frame() sur des trames complètes."""

    def test_empty_frame_returns_no_targets(self, empty_frame_bytes: bytes) -> None:
        """Une trame avec tous les slots à zéro doit donner 0 cible."""
        frame = parse_frame(empty_frame_bytes)
        assert frame.target_count == 0
        assert frame.has_targets is False
        assert frame.targets == []

    def test_one_target_frame(self, one_target_frame_bytes: bytes) -> None:
        """Une trame avec 1 cible doit donner exactement 1 cible."""
        frame = parse_frame(one_target_frame_bytes)
        assert frame.target_count == 1
        assert frame.targets[0].slot == 0

    def test_three_target_frame(self, three_target_frame_bytes: bytes) -> None:
        """Une trame avec 3 cibles doit donner 3 cibles dans les bons slots."""
        frame = parse_frame(three_target_frame_bytes)
        assert frame.target_count == 3
        # Les slots doivent être 0, 1, 2 dans l'ordre.
        slots = [t.slot for t in frame.targets]
        assert slots == [0, 1, 2]

    def test_raw_hex_only_when_requested(self, one_target_frame_bytes: bytes) -> None:
        """raw_hex doit être None par défaut, rempli si include_raw=True."""
        frame_default = parse_frame(one_target_frame_bytes)
        assert frame_default.raw_hex is None

        frame_with_raw = parse_frame(one_target_frame_bytes, include_raw=True)
        assert frame_with_raw.raw_hex is not None
        assert len(frame_with_raw.raw_hex) > 0


class TestParseFrameErrors:
    """Tests de robustesse aux trames malformées."""

    def test_wrong_size_raises(self) -> None:
        """Une trame trop courte ou trop longue doit lever LD2450ParseError."""
        with pytest.raises(LD2450ParseError, match="Taille"):
            parse_frame(b"\x00" * 29)  # trop court
        with pytest.raises(LD2450ParseError, match="Taille"):
            parse_frame(b"\x00" * 31)  # trop long

    def test_invalid_header_raises(self) -> None:
        """Un header corrompu doit lever LD2450ParseError."""
        bad_frame = bytes([0xBB, 0xFF, 0x03, 0x00]) + bytes(24) + FRAME_FOOTER
        with pytest.raises(LD2450ParseError, match="Header"):
            parse_frame(bad_frame)

    def test_invalid_footer_raises(self) -> None:
        """Un footer corrompu doit lever LD2450ParseError."""
        bad_frame = FRAME_HEADER + bytes(24) + bytes([0xAA, 0xBB])
        with pytest.raises(LD2450ParseError, match="Footer"):
            parse_frame(bad_frame)

    def test_empty_bytes_raises(self) -> None:
        """Une chaîne vide doit lever LD2450ParseError."""
        with pytest.raises(LD2450ParseError):
            parse_frame(b"")