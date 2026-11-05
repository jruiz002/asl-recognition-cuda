"""Utilidades compartidas por la versión secuencial y la paralela."""

import sys


def consola_utf8() -> None:
    """Evita UnicodeEncodeError en consolas Windows con cp1252 (símbolos como ✔, →, ×)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
