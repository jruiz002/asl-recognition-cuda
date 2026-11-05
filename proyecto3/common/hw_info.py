"""Registro del hardware y software de cada ejecución (evidencia de 'mismas condiciones')."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _cpu_name() -> str:
    """Nombre comercial del CPU (Windows/macOS/Linux)."""
    try:
        if platform.system() == "Windows":
            output = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if output:
                return output
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "desconocido"


def _nvidia_smi() -> dict[str, str] | None:
    """Driver y GPU según nvidia-smi (None si no existe)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    name, driver, memory = [part.strip() for part in output.splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "memory_total": memory}


def collect_hw_info() -> dict:
    import psutil
    import torch

    info: dict = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "sistema": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True) or os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "torch": torch.__version__,
        "torch_hilos_intraop": torch.get_num_threads(),
        "cuda_disponible": torch.cuda.is_available(),
        "cuda_version_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numba": _version("numba"),
        "numba_cuda": _version("numba-cuda"),
        "numpy": _version("numpy"),
        "pillow": _version("Pillow"),
        "pandas": _version("pandas"),
        "psutil": _version("psutil"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_memoria_gb=round(props.total_memory / 1024**3, 1),
            gpu_sms=props.multi_processor_count,
            gpu_compute_capability=f"{props.major}.{props.minor}",
        )
    smi = _nvidia_smi()
    if smi:
        info.update(driver_nvidia=smi["driver"], gpu_smi=smi["name"])
    return info
