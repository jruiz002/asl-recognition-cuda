"""Configuración del benchmark del Proyecto 3.

Todos los hiperparámetros se toman del laboratorio original para que la versión
secuencial y la paralela trabajen con los MISMOS datos y las MISMAS condiciones:

- Partición: ``data/manifests/all.csv`` (17 400 imágenes, 70/15/15, semilla 42).
- Resolución 64×64 RGB, batch 64, AdamW con weight_decay=1e-4 (``src/neural.py``).
- Experimentos: los definidos en ``entrenar_modelos.NEURAL_EXPERIMENTS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Raíz del repositorio (este archivo vive en proyecto3/common/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Experimentos del lab seleccionados para el benchmark (justificación en
# docs/decisiones_alcance.md):
#   - cnn_reg_drop035: CNN más pesada en convoluciones → mejor caso para la GPU.
#   - cnn_base_aug:    augmentation por imagen en cada época → muestra el trabajo
#                      del CPU multinúcleo (DataLoader con workers).
DEFAULT_EXPERIMENTS = ("cnn_reg_drop035", "cnn_base_aug")


@dataclass
class BenchmarkConfig:
    """Parámetros de una corrida del benchmark (iguales para ambas versiones)."""

    experiments: tuple[str, ...] = DEFAULT_EXPERIMENTS
    epochs: int = 6               # igual que el lab (entrenar_modelos.py --epochs 6)
    batch_size: int = 64          # igual que el lab
    image_size: int = 64          # igual que el lab
    seed: int = 42                # igual que el lab
    runs: int = 5                 # repeticiones medidas (sin contar el warm-up)
    warmup: bool = True           # corrida de calentamiento no contabilizada
    sample_per_class: int | None = None   # modo prueba rápida (--sample N)
    threads: int | None = None    # None = igual que el lab: min(8, hilos de torch)
    workers: int | None = None    # procesos CPU de la versión paralela (None = todos los núcleos)
    manifest: Path = ROOT / "data" / "manifests" / "all.csv"
    data_root: Path = ROOT / "data" / "raw"
    results_dir: Path = ROOT / "results"
    evidence_dir: Path = ROOT / "evidence"
    work_dir: Path = ROOT / "data" / "bench_cache"   # cachés temporales por corrida
    extra: dict = field(default_factory=dict)

    @property
    def patience(self) -> int:
        """Early stopping desactivado: patience > epochs → siempre se corren todas las épocas.

        El lab usa patience=2; con épocas fijas ambas versiones hacen exactamente
        el mismo trabajo y los tiempos son comparables (docs/decisiones_alcance.md).
        """
        return self.epochs + 1
