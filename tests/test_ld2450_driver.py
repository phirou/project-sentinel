"""
Tests unitaires pour le driver UART LD2450.

Ces tests valident la logique de synchronisation header/footer SANS
nécessiter de hardware réel. On teste directement la méthode interne
_try_extract_frame() avec des buffers fabriqués couvrant tous les
cas de figure.

Le test d'intégration avec un vrai port série est marqué @pytest.mark.hardware
et skip par défaut en CI.
"""

from __future__ import annotations

import pytest

from sentinel.drivers.ld2450_driver import LD2450Driver
from sentinel.drivers.ld2450_parser import FRAME_FOOTER, FRAME_HEADER, FRAME_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES DE TEST
# ─────────────────────────────────────────────────────────────────────────────

def make_valid_frame(payload_filler: int = 0x42) -> bytes:
    """
    Construit une trame valide (header + 24 octets quelconques + footer).

    Les 24 octets de "payload" n'ont pas besoin d'être un vrai contenu
    radar : on teste juste l'extraction, pas le parsing. Le contenu est
    rempli avec la valeur payload_filler pour faciliter le debug visuel.
    """
    payload = bytes([payload_filler] * 24)
    frame = FRAME_HEADER + payload + FRAME_FOOTER
    assert len(frame) == FRAME_SIZE
    return frame


@pytest.fixture
def driver() -> LD2450Driver:
    """Instance de driver pour les tests (port fictif, on ne l'ouvrira pas)."""
    return LD2450Driver(port="/dev/null", baudrate=256000)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE SYNCHRONISATION
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameExtraction:
    """Tests de la méthode _try_extract_frame()."""

    def test_extracts_valid_frame_at_start_of_buffer(
        self, driver: LD2450Driver
    ) -> None:
        """Buffer commençant par une trame valide : extraction immédiate."""
        frame = make_valid_frame()
        buffer = bytearray(frame)

        result = driver._try_extract_frame(buffer)

        assert result == bytearray(frame)
        assert len(buffer) == 0  # le buffer doit être vidé après extraction

    def test_returns_none_if_buffer_too_short(
        self, driver: LD2450Driver
    ) -> None:
        """Si le buffer contient le header mais < 30 octets, on attend."""
        partial = FRAME_HEADER + bytes(10)  # 14 octets, pas assez
        buffer = bytearray(partial)

        result = driver._try_extract_frame(buffer)

        assert result is None
        # Le buffer doit être conservé : on attend plus de données.
        assert len(buffer) == 14

    def test_skips_noise_before_header(self, driver: LD2450Driver) -> None:
        """Octets parasites avant le header : ils doivent être jetés."""
        noise = bytes([0x11, 0x22, 0x33, 0x44, 0x55])
        frame = make_valid_frame()
        buffer = bytearray(noise + frame)

        result = driver._try_extract_frame(buffer)

        assert result == bytearray(frame)
        assert len(buffer) == 0

    def test_returns_none_if_no_header(self, driver: LD2450Driver) -> None:
        """Buffer sans aucun header : on attend, on ne consomme presque rien."""
        # 50 octets de bruit qui ne contiennent pas AA FF 03 00.
        noise = bytes([0x11, 0x22, 0x33] * 17)
        buffer = bytearray(noise)
        initial_len = len(buffer)

        result = driver._try_extract_frame(buffer)

        assert result is None
        # On doit garder seulement quelques octets de fin (au cas où le
        # header commencerait dans les derniers octets, à compléter au
        # prochain chunk).
        assert len(buffer) < initial_len
        assert len(buffer) <= len(FRAME_HEADER)

    def test_rejects_false_positive_header(self, driver: LD2450Driver) -> None:
        """
        Header trouvé mais footer absent : c'est un faux positif (le pattern
        AA FF 03 00 est apparu par hasard dans des données). On doit avancer
        d'1 octet et chercher plus loin.
        """
        # Buffer qui contient AA FF 03 00 mais pas de footer 55 CC à la bonne
        # position, suivi d'une vraie trame valide.
        false_positive = FRAME_HEADER + bytes(26)  # 30 octets sans footer correct
        # Mais le 28ème octet doit être différent de 0x55 pour être un faux.
        # bytes(26) donne 26 zéros, donc footer = 00 00 → invalide. OK.
        real_frame = make_valid_frame()
        buffer = bytearray(false_positive + real_frame)

        # Premier appel : doit échouer sur le faux positif, avancer dans le buffer.
        result1 = driver._try_extract_frame(buffer)
        assert result1 is None

        # Appels successifs : on doit finir par trouver la vraie trame.
        # On boucle parce que chaque appel n'avance que d'un octet en cas
        # de faux positif.
        for _ in range(100):  # garde-fou anti-boucle infinie
            result = driver._try_extract_frame(buffer)
            if result is not None:
                break
        else:
            pytest.fail("Le driver n'a pas réussi à trouver la trame valide")

        assert result == bytearray(real_frame)

    def test_extracts_multiple_consecutive_frames(
        self, driver: LD2450Driver
    ) -> None:
        """Deux trames collées : doivent être extraites successivement."""
        frame1 = make_valid_frame(payload_filler=0xAA)
        frame2 = make_valid_frame(payload_filler=0xBB)
        buffer = bytearray(frame1 + frame2)

        # Première extraction
        result1 = driver._try_extract_frame(buffer)
        assert result1 == bytearray(frame1)
        assert len(buffer) == FRAME_SIZE  # il reste frame2

        # Seconde extraction
        result2 = driver._try_extract_frame(buffer)
        assert result2 == bytearray(frame2)
        assert len(buffer) == 0


class TestDriverStats:
    """Tests des compteurs internes du driver."""

    def test_initial_stats_are_zero(self, driver: LD2450Driver) -> None:
        stats = driver.stats
        assert stats["frames_emitted"] == 0
        assert stats["invalid_footers"] == 0
        assert stats["reconnect_count"] == 0

    def test_invalid_footer_increments_counter(
        self, driver: LD2450Driver
    ) -> None:
        """Un faux positif doit incrémenter le compteur d'invalid_footers."""
        # 30 octets avec header mais sans footer valide.
        bad_frame = FRAME_HEADER + bytes(26)
        buffer = bytearray(bad_frame)

        driver._try_extract_frame(buffer)
        assert driver.stats["invalid_footers"] == 1