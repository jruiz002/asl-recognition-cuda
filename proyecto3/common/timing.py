"""Medición de tiempos por fase.

Reglas aplicadas (docs/02_medicion_secuencial.md):

- Reloj de pared con ``time.perf_counter()`` (monótono y de alta resolución).
- Las operaciones CUDA son ASÍNCRONAS: la CPU encola kernels y sigue. Por eso,
  antes de abrir y de cerrar cada fase se llama a ``torch.cuda.synchronize()``,
  así el tiempo incluye el trabajo que la GPU realmente terminó.
- Para sub-fases que ocurren dentro del bucle de entrenamiento (kernel por lote,
  copia H2D por lote) se usan ``torch.cuda.Event``, que miden en la propia GPU sin
  forzar sincronizaciones en cada lote (no alteran el rendimiento medido).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch

# Nombres canónicos de las fases (columnas del CSV de resultados).
PHASES = (
    "preparacion_cpu",      # FASE 1: lectura, decodificación JPEG, resize → uint8
    "transferencia_h2d",    # copia CPU → GPU (solo versión paralela)
    "entrenamiento",        # forward + backward + optimizador (todas las épocas)
    "evaluacion",           # validación por época + prueba final
    "total",
)
# Sub-métricas informativas (ya incluidas dentro de "entrenamiento"):
#   carga_lotes_cpu      = tiempo esperando el siguiente lote del DataLoader (trabajo de CPU)
#   kernel_normalizacion = kernel Numba uint8→float32 en la GPU (eventos CUDA)
#   h2d_en_bucle         = copias CPU→GPU por lote (solo experimento con augmentation)
SUBMETRICS = ("carga_lotes_cpu", "kernel_normalizacion", "h2d_en_bucle")


def cuda_sync() -> None:
    """Sincroniza la GPU si existe; en CPU no hace nada."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseTimer:
    """Acumula segundos por fase. Uso: ``with timer.fase("entrenamiento"): ...``."""

    def __init__(self, on_phase=None) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self._on_phase = on_phase   # callback opcional (etiqueta el monitor de CPU/GPU)

    @contextmanager
    def fase(self, name: str):
        cuda_sync()
        if self._on_phase:
            self._on_phase(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.seconds[name] += time.perf_counter() - started
            if self._on_phase:
                self._on_phase("idle")

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] += seconds

    def as_dict(self) -> dict[str, float]:
        return {name: round(self.seconds.get(name, 0.0), 6) for name in (*PHASES, *SUBMETRICS)}


class CudaEventAccumulator:
    """Acumula duraciones medidas con eventos CUDA (en milisegundos de GPU).

    ``start()``/``stop()`` solo registran eventos en el stream; el cálculo se hace
    en ``total_seconds()`` tras sincronizar, una vez por época.
    """

    def __init__(self) -> None:
        self.enabled = torch.cuda.is_available()
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._current = None
        self._accumulated_ms = 0.0

    def start(self) -> None:
        if self.enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._current = event

    def stop(self) -> None:
        if self.enabled and self._current is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._pairs.append((self._current, event))
            self._current = None

    def collect(self) -> None:
        """Convierte los pares pendientes a milisegundos (requiere sincronizar)."""
        if not self._pairs:
            return
        torch.cuda.synchronize()
        self._accumulated_ms += sum(start.elapsed_time(end) for start, end in self._pairs)
        self._pairs.clear()

    def total_seconds(self) -> float:
        self.collect()
        return self._accumulated_ms / 1000.0
