"""Empaqueta SOLO las 17 400 imágenes del manifiesto en un ZIP para llevarlo en USB.

El dataset completo pesa ~1,2 GB (87 000 imágenes); el proyecto usa 17 400. El ZIP
conserva la estructura que espera el código:

    asl_alphabet_train/asl_alphabet_train/<etiqueta>/<archivo>.jpg

Uso (en la Mac):   python scripts/export_subset.py           → asl_subset.zip
En la laptop:      descomprimir dentro de data\\raw\\  (queda data\\raw\\asl_alphabet_train\\...)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proyecto3.common.paths import load_portable_manifest  # noqa: E402


def main() -> int:
    data = load_portable_manifest(ROOT / "data" / "manifests" / "all.csv", ROOT / "data" / "raw")
    output = ROOT / "asl_subset.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:   # JPEG ya está comprimido
        for path, label in zip(data["path"], data["label"]):
            archive.write(path, f"asl_alphabet_train/asl_alphabet_train/{label}/{Path(path).name}")
    print(f"{len(data):,} imágenes → {output} ({output.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
