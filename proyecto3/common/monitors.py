"""Monitores de recursos en segundo plano (evidencia para la rúbrica: 30 pts).

- ``NvidiaSmiLogger``: lanza ``nvidia-smi ... -l 1`` y guarda un CSV con el uso de
  la GPU (núcleos y memoria) cada segundo durante todo el benchmark.
- ``CpuPerCoreMonitor``: hilo que muestrea ``psutil.cpu_percent(percpu=True)`` y
  etiqueta cada muestra con la versión, el experimento y la FASE activa. Así se puede
  demostrar que la preparación usó todos los núcleos y que el entrenamiento en GPU
  dejó libre al CPU.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


class NvidiaSmiLogger:
    QUERY = (
        "timestamp,name,utilization.gpu,utilization.memory,memory.used,"
        "memory.total,temperature.gpu,power.draw"
    )

    def __init__(self, output: Path, interval_s: int = 1) -> None:
        self.output = output
        self.interval_s = interval_s
        self._process: subprocess.Popen | None = None
        self._handle = None

    @property
    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def start(self) -> bool:
        if not self.available:
            print("[monitor] nvidia-smi no encontrado: no se registrará la GPU.")
            return False
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv",
             "-l", str(self.interval_s)],
            stdout=self._handle, stderr=subprocess.DEVNULL,
        )
        print(f"[monitor] nvidia-smi registrando en {self.output}")
        return True

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._handle is not None:
            self._handle.close()


class CpuPerCoreMonitor:
    """Muestrea el uso de cada núcleo lógico y la fase activa del pipeline."""

    def __init__(self, output: Path, interval_s: float = 0.25) -> None:
        import psutil

        self._psutil = psutil
        self.output = output
        self.interval_s = interval_s
        self.cores = psutil.cpu_count(logical=True)
        self.context = {"version": "-", "experimento": "-", "corrida": "-"}
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_context(self, **values) -> None:
        self.context.update({key: str(value) for key, value in values.items()})

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._psutil.cpu_percent(percpu=True)   # primera llamada: inicializa contadores
        self._thread.start()
        print(f"[monitor] uso de CPU por núcleo en {self.output}")

    def _run(self) -> None:
        header = ["timestamp", "t_rel_s", "version", "experimento", "corrida", "fase",
                  "cpu_total", *[f"nucleo_{index}" for index in range(self.cores)]]
        started = time.perf_counter()
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            while not self._stop.wait(self.interval_s):
                per_core = self._psutil.cpu_percent(percpu=True)
                writer.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    round(time.perf_counter() - started, 2),
                    self.context["version"], self.context["experimento"],
                    self.context["corrida"], self.phase,
                    round(sum(per_core) / len(per_core), 1), *per_core,
                ])
                handle.flush()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
