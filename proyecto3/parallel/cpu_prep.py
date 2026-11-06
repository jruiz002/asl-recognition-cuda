"""FASE 1 – PREPARACIÓN DE DATOS EN CPU MULTINÚCLEO.

Aquí está el manejo de archivos (lectura de disco + decodificación JPEG + resize).
Es trabajo que la GPU NO puede acelerar de forma útil:

- La lectura de disco y las llamadas al sistema operativo solo ocurren en el CPU.
- Decodificar JPEG (Huffman + IDCT por bloques) es una tarea ramificada y
  secuencial dentro de cada archivo, pero cada archivo es INDEPENDIENTE de los demás.
  Eso es paralelismo de datos "vergonzosamente paralelo" a nivel de archivo, ideal
  para repartir entre los núcleos del CPU.

Diseño:
- ``ProcessPoolExecutor`` con un proceso por núcleo lógico. Se usan PROCESOS (no
  hilos) porque el GIL de Python serializaría el trabajo de los hilos.
- Las imágenes se reparten en bloques de ``CHUNK_SIZE`` (balance de carga dinámico:
  el proceso que termina primero toma el siguiente bloque).
- Cada proceso escribe sus imágenes DIRECTAMENTE en un archivo ``.npy`` memmap
  compartido. Así se evita devolver los píxeles por pickle al proceso principal.
- El resultado es byte a byte igual al caché del lab (``build_image_cache``), ya que
  se usan exactamente las mismas operaciones de Pillow (lo verifica tests/test_proyecto3.py).

IMPORTANTE (Windows): los procesos se crean con "spawn", que reimporta los
módulos. Por eso este archivo solo importa NumPy y Pillow (NO torch) y las
funciones de trabajo están a nivel de módulo (deben poder serializarse con pickle).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CHUNK_SIZE = 128   # imágenes por tarea: ~136 tareas para 17 400 imágenes


def default_workers() -> int:
    """Un proceso por núcleo lógico del CPU."""
    return os.cpu_count() or 1


def _decode_resize(path: str, image_size: int) -> np.ndarray:
    """Las mismas operaciones que build_image_cache (src/neural.py)."""
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.uint8)


def _process_chunk(task: tuple[str, int, list[str], int]) -> int:
    """Trabajo de UN proceso: decodifica un bloque y lo escribe en el memmap compartido."""
    cache_path, start, paths, image_size = task
    cache = np.load(cache_path, mmap_mode="r+")
    for offset, path in enumerate(paths):
        cache[start + offset] = _decode_resize(path, image_size)
    cache.flush()
    del cache
    return len(paths)


def prepare_splits_parallel(
    split_paths: dict[str, list[str]],
    cache_dir: Path,
    image_size: int = 64,
    workers: int | None = None,
) -> dict[str, Path]:
    """Decodifica todas las particiones en paralelo y devuelve la ruta de cada caché uint8.

    Un solo pool atiende las tres particiones (train/validation/test), así ningún
    núcleo queda ocioso entre particiones.
    """
    workers = workers or default_workers()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, int, list[str], int]] = []
    outputs: dict[str, Path] = {}
    for split, paths in split_paths.items():
        cache_path = cache_dir / f"{split}_{image_size}.npy"
        # El proceso principal solo reserva el archivo (cabecera + tamaño).
        np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(paths), image_size, image_size, 3),
        ).flush()
        outputs[split] = cache_path
        for start in range(0, len(paths), CHUNK_SIZE):
            tasks.append((str(cache_path), start, paths[start : start + CHUNK_SIZE], image_size))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        processed = sum(pool.map(_process_chunk, tasks))
    expected = sum(len(paths) for paths in split_paths.values())
    if processed != expected:
        raise RuntimeError(f"Se procesaron {processed} imágenes de {expected}")
    return outputs
