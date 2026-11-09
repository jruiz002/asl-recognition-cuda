"""Arma el documento ÚNICO de evidencia de ejecución (rúbrica: 30 pts).

Centraliza en ``evidence/EVIDENCIA.md`` todo lo que demuestra que el entorno
reconoció y usó el hardware asignado EN CADA FASE:

    1. Hardware y software detectados      (results/hw_info.json, evidence/check_env.txt)
    2. Uso de hardware por fase             (cruce de evidence/cpu_por_nucleo_<stamp>.csv
                                             con evidence/nvidia_smi_<stamp>.csv)
    3. Extractos de los logs                (results/logs/benchmark_<stamp>.log)
    4. Salidas de consola guardadas         (evidence/consola_*.txt)
    5. Capturas de pantalla                 (evidence/capturas/*.png)
    6. Figuras generadas a partir de los logs (results/figuras/*.png)
    7. Tiempos y speedup                    (results/tabla_speedup.md, kernel_benchmark.md)

Uso (después del benchmark oficial y de tomar las capturas):

    python scripts/armar_evidencia.py            # usa el benchmark más reciente
    python scripts/armar_evidencia.py --stamp 20261107_190000

No ejecuta nada en la GPU: solo lee archivos ya generados.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
RESULTS = ROOT / "results"
OUTPUT = EVIDENCE / "EVIDENCIA.md"

PHASE_LABELS = {
    "preparacion_cpu": "FASE 1 · Preparación de imágenes",
    "transferencia_h2d": "Transferencia CPU → GPU",
    "entrenamiento": "FASE 2 · Entrenamiento",
    "evaluacion": "FASE 2 · Evaluación",
}
# Hardware que DEBERÍA usar cada (versión, fase) según el diseño del pipeline.
EXPECTED = {
    ("secuencial", "preparacion_cpu"): "CPU, 1 proceso",
    ("secuencial", "entrenamiento"): "CPU (hilos de PyTorch), GPU sin uso",
    ("secuencial", "evaluacion"): "CPU, GPU sin uso",
    ("paralela", "preparacion_cpu"): "CPU, todos los núcleos",
    ("paralela", "transferencia_h2d"): "Bus PCIe (CPU → GPU)",
    ("paralela", "entrenamiento"): "GPU (CUDA); CPU libre o con workers de augmentation",
    ("paralela", "evaluacion"): "GPU (CUDA)",
}

# Capturas esperadas (nombre → qué demuestran). Ver GUIA_EJECUCION.md.
SCREENSHOTS = [
    ("01_nvidia_smi_inicial.png", "Hardware", "nvidia-smi antes de empezar: GPU, driver y versión de CUDA reconocidos."),
    ("02_check_env.png", "Hardware", "check_env.py: PyTorch y Numba ven la GPU y el CPU de 32 hilos; dataset listo."),
    ("03_cpu_preparacion_secuencial.png", "FASE 1 · secuencial", "Preparación del lab original: un solo proceso, uso total del CPU bajo."),
    ("04_cpu_preparacion_paralela.png", "FASE 1 · paralela", "Preparación paralela: los 32 procesadores lógicos cargados."),
    ("05_gpu_entrenamiento_taskmgr.png", "FASE 2 · GPU", "Entrenamiento: gráfico Cuda alto y memoria dedicada de la GPU ocupada."),
    ("06_cpu_durante_gpu.png", "FASE 2 · GPU", "CPU relativamente libre mientras la GPU entrena."),
    ("07_nvidia_smi_entrenamiento.png", "FASE 2 · GPU", "nvidia-smi durante el entrenamiento: GPU-Util alto y python.exe usando la GPU."),
    ("08_bench_kernel.png", "Kernel CUDA", "Kernel Numba propio vs NumPy, verificado bit a bit."),
    ("09_secuencial_entrenamiento_cpu.png", "Benchmark · secuencial", "Entrenamiento secuencial en el CPU; la GPU NVIDIA sin uso."),
    ("10_paralela_entrenamiento_gpu.png", "Benchmark · paralela (capturas 10 y 11 unificadas)",
     "Benchmark oficial, versión paralela de cnn_reg_drop035 (corridas 3–5). En una sola imagen: "
     "nvidia-smi -l 2 con GPU-Util 98 %, estado P0, 83 W, 527 MiB y python3.12.exe como proceso tipo C (CUDA); "
     "el log con 'GPU detectada', 'preparación con 32 procesos de CPU' y los tiempos por fase; y el "
     "Administrador de tareas con los 32 procesadores lógicos mostrando una ráfaga periódica por cada "
     "corrida (FASE 1 en el CPU) mientras la GPU entrena entre ráfagas (FASE 2)."),
    ("12_paralela_aug_cpu_y_gpu.png", "Benchmark · paralela",
     "cnn_base_aug durante el entrenamiento (épocas 1–5): GPU en estado P0 con python3.12.exe como proceso "
     "CUDA (tipo C) y los workers de augmentation repartidos en los núcleos del CPU. El GPU-Util instantáneo "
     "es bajo (4 %): con augmentation la GPU ESPERA a que el CPU genere los lotes. En los registros de "
     "nvidia-smi el GPU-Util medio al entrenar es ~14–18 % en este modelo contra ~76 % en cnn_reg_drop035 "
     "(sin augmentation): el cuello de botella es la preparación en el CPU, lo que explica su menor speedup. "
     "Captura tomada en una corrida extra de 12 épocas (no incluida en los promedios) para tener tiempo de capturar."),
    ("13_resumen_speedup.png", "Resultado", "Tabla RESUMEN de speedup al final del benchmark."),
]

FIGURES = [
    ("01_tiempos_por_fase.png", "Tiempo medio por fase, secuencial vs paralela."),
    ("02_speedup_por_fase.png", "Speedup por fase (t_secuencial / t_paralela)."),
    ("03_cpu_preparacion_por_nucleo.png", "Uso medio de cada núcleo durante la FASE 1 (secuencial vs paralela)."),
    ("03_cpu_nucleos_secuencial.png", "Mapa de calor del uso por núcleo a lo largo de la versión secuencial."),
    ("03_cpu_nucleos_paralela.png", "Mapa de calor del uso por núcleo a lo largo de la versión paralela."),
    ("04_gpu_uso.png", "Utilización y memoria de la GPU (nvidia-smi) durante todo el benchmark."),
    ("05_kernel.png", "Kernel CUDA propio vs NumPy."),
]

LOG_PATTERNS = re.compile(
    r"(=====|SECUENCIAL ·|PARALELA ·|GPU detectada|AVISO|procesos de CPU|workers de CPU|"
    r"tiempos \(s\)|hilos|Traceback|Error)"
)


def rel(path: Path) -> str:
    """Ruta relativa a evidence/ con barras normales (para que el .md muestre las imágenes)."""
    return Path("..", path.relative_to(ROOT)).as_posix() if EVIDENCE not in path.parents else \
        path.relative_to(EVIDENCE).as_posix()


def find_stamps(requested: list[str] | None) -> list[str]:
    """Marcas de tiempo a incluir: las pedidas o TODOS los benchmarks de results/logs/.

    El benchmark oficial puede quedar repartido en varias ejecuciones (p. ej. si una se
    interrumpió y se completó después sin ``--fresh``); se juntan todas.
    """
    if requested:
        return requested
    logs = sorted((RESULTS / "logs").glob("benchmark_*.log"))
    return [log.stem.removeprefix("benchmark_") for log in logs]


def read_text(path: Path) -> str:
    """Lee UTF-8 o UTF-16 (``Tee-Object`` de PowerShell 5.1 guarda en UTF-16 con BOM)."""
    if not path.exists():
        return ""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


# ------------------------- Cruce CPU por núcleo + GPU --------------------------
def phase_hardware_table(stamps: list[str]) -> str:
    """Tabla de uso de hardware por fase, juntando los CSV de todos los ``stamps``."""
    blocks = [phase_hardware_table_one(stamp) for stamp in stamps]
    return "\n".join(f"#### Ejecución `{stamp}`\n\n{block}" for stamp, block in zip(stamps, blocks))


HEADER = re.compile(r"^(SECUENCIAL|PARALELA)\S* · (\S+) · corrida (\d+)/\d+")


def completed_runs(stamp: str) -> set[tuple[str, str, str]]:
    """(versión, experimento, corrida) que el log marca como TERMINADAS (tienen su línea 'tiempos (s)').

    Así se excluyen las muestras de una corrida interrumpida (p. ej. por falta de memoria),
    que no llegó a ``tiempos.csv`` y no forma parte de los promedios.
    """
    done, current = set(), None
    for line in read_text(RESULTS / "logs" / f"benchmark_{stamp}.log").splitlines():
        match = HEADER.match(line.strip())
        if match:
            current = (match.group(1).lower(), match.group(2), match.group(3))
        elif "tiempos (s)" in line and current:
            done.add(current)
            current = None
    return done


def phase_hardware_table_one(stamp: str) -> str:
    cpu_path = EVIDENCE / f"cpu_por_nucleo_{stamp}.csv"
    gpu_path = EVIDENCE / f"nvidia_smi_{stamp}.csv"
    if not cpu_path.exists():
        return f"_No existe `{cpu_path.name}`._\n"

    cpu = pd.read_csv(cpu_path, dtype={"corrida": str})
    cpu["timestamp"] = pd.to_datetime(cpu["timestamp"])
    core_cols = [column for column in cpu.columns if column.startswith("nucleo_")]
    cpu["nucleos_50"] = (cpu[core_cols] > 50).sum(axis=1)
    done = completed_runs(stamp)
    keep = pd.Series(
        [(v.split("_")[0], e, c) in done for v, e, c in zip(cpu["version"], cpu["experimento"], cpu["corrida"])],
        index=cpu.index,
    )
    cpu = cpu[keep & cpu["fase"].isin(PHASE_LABELS)]
    if cpu.empty:
        return "_No hay muestras de corridas medidas en el CSV de CPU._\n"

    has_gpu = gpu_path.exists()
    if has_gpu:
        gpu = pd.read_csv(gpu_path, skipinitialspace=True)
        gpu.columns = [column.strip() for column in gpu.columns]
        gpu = gpu[gpu["timestamp"] != "timestamp"]   # nvidia-smi repite la cabecera a veces
        gpu["timestamp"] = pd.to_datetime(gpu["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
        util_col = next(c for c in gpu.columns if c.startswith("utilization.gpu"))
        mem_col = next(c for c in gpu.columns if c.startswith("memory.used"))
        gpu["gpu_util"] = gpu[util_col].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        gpu["gpu_mem"] = gpu[mem_col].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        cpu = pd.merge_asof(
            cpu.sort_values("timestamp"), gpu[["timestamp", "gpu_util", "gpu_mem"]].sort_values("timestamp"),
            on="timestamp", direction="nearest", tolerance=pd.Timedelta("1.5s"),
        )

    lines = [
        f"Fuente: `evidence/{cpu_path.name}` (uso de los {len(core_cols)} procesadores lógicos cada 0,25 s, "
        "etiquetado con la fase activa)" + (f" cruzado por hora con `evidence/{gpu_path.name}` "
        "(nvidia-smi cada 1 s)." if has_gpu else "."),
        "Solo corridas medidas y terminadas (sin calentamiento ni corridas interrumpidas). "
        "Núcleos activos = procesadores lógicos con uso > 50 %.",
        "",
        "| Versión | Modelo | Fase | Hardware esperado | CPU total medio | Núcleos activos (medio / máx) "
        "| GPU-Util medio / máx | Memoria GPU máx |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    order = list(PHASE_LABELS)
    grouped = cpu.groupby(["version", "experimento", "fase"], sort=False)
    rows = sorted(grouped, key=lambda item: (item[0][1], item[0][0] != "secuencial", order.index(item[0][2])))
    for (version, experiment, phase), frame in rows:
        expected = EXPECTED.get((version.split("_")[0], phase), "–")
        gpu_cells = "– | –"
        if has_gpu and frame["gpu_util"].notna().any():
            gpu_cells = (f"{frame['gpu_util'].mean():.0f} % / {frame['gpu_util'].max():.0f} % | "
                         f"{frame['gpu_mem'].max():,.0f} MiB")
        lines.append(
            f"| {version} | {experiment} | {PHASE_LABELS[phase]} | {expected} | {frame['cpu_total'].mean():.1f} % "
            f"| {frame['nucleos_50'].mean():.1f} / {frame['nucleos_50'].max()} | {gpu_cells} |"
        )
    lines.append("")
    lines.append(
        "> Nota: en fases muy cortas (p. ej. la transferencia H2D, < 0,1 s) puede no haber muestras; "
        "nvidia-smi muestrea cada 1 s, así que en fases de pocos segundos el GPU-Util es aproximado. "
        "La memoria de la GPU es la de TODA la tarjeta (incluye unos cientos de MiB del escritorio de "
        "Windows y del contexto CUDA que queda de corridas anteriores); lo que demuestra el uso de CUDA es el GPU-Util."
    )
    return "\n".join(lines) + "\n"


def log_excerpt(stamps: list[str]) -> str:
    return "\n".join(log_excerpt_one(stamp) for stamp in stamps)


def log_excerpt_one(stamp: str) -> str:
    path = RESULTS / "logs" / f"benchmark_{stamp}.log"
    if not path.exists():
        return f"_No existe `{path.name}`._\n"
    selected = [line.rstrip() for line in read_text(path).splitlines()
                if LOG_PATTERNS.search(line) and not set(line.strip()) <= {"="}]
    body = "\n".join(selected[:200])
    return (f"Log completo: [`results/logs/{path.name}`]({rel(path)}). "
            "Extracto (detección de hardware, procesos usados y tiempos por corrida):\n\n"
            f"```text\n{body}\n```\n")


def code_block(path: Path, title: str) -> str:
    text = read_text(path).strip()
    if not text:
        return f"_Falta `{path.relative_to(ROOT).as_posix()}`._\n"
    return f"**{title}** · [`{path.relative_to(ROOT).as_posix()}`]({rel(path)})\n\n```text\n{text}\n```\n"


def hardware_section() -> str:
    path = RESULTS / "hw_info.json"
    if not path.exists():
        return "_Falta `results/hw_info.json`._\n"
    info = json.loads(read_text(path))
    keys = [
        ("cpu", "CPU"), ("nucleos_fisicos", "Núcleos físicos"), ("nucleos_logicos", "Procesadores lógicos"),
        ("ram_gb", "RAM (GB)"), ("gpu", "GPU"), ("gpu_memoria_gb", "Memoria GPU (GB)"),
        ("gpu_sms", "Multiprocesadores (SM)"), ("gpu_compute_capability", "Compute capability"),
        ("driver_nvidia", "Driver NVIDIA"), ("cuda_version_torch", "CUDA (PyTorch)"), ("cudnn", "cuDNN"),
        ("torch", "PyTorch"), ("numba", "Numba"), ("numba_cuda", "numba-cuda"), ("numpy", "NumPy"),
        ("python", "Python"), ("sistema", "Sistema operativo"),
    ]
    rows = ["| Componente | Valor |", "|---|---|"]
    rows += [f"| {label} | {info[key]} |" for key, label in keys if key in info]
    return "\n".join(rows) + "\n"


def screenshots_section() -> str:
    folder = EVIDENCE / "capturas"
    out, missing = [], []
    for name, group, caption in SCREENSHOTS:
        path = folder / name
        if path.exists():
            out.append(f"#### {name.removesuffix('.png')} · {group}\n\n{caption}\n\n![{name}](capturas/{name})\n")
        else:
            missing.append(name)
    extra = sorted(p.name for p in folder.glob("*.png") if p.name not in {n for n, _, _ in SCREENSHOTS})
    for name in extra:
        out.append(f"#### {name.removesuffix('.png')} · adicional\n\n![{name}](capturas/{name})\n")
    status = f"Capturas incluidas: **{len(SCREENSHOTS) - len(missing)} de {len(SCREENSHOTS)}**."
    if missing:
        status += " Faltan: " + ", ".join(f"`{m}`" for m in missing) + "."
    return status + "\n\n" + "\n".join(out)


def figures_section() -> str:
    out = []
    for name, caption in FIGURES:
        path = RESULTS / "figuras" / name
        if path.exists():
            out.append(f"**{caption}**\n\n![{name}]({rel(path)})\n")
    return "\n".join(out) or "_Aún no hay figuras en `results/figuras/`._\n"


def console_section() -> str:
    files = sorted(EVIDENCE.glob("consola_*.txt"))
    if not files:
        return "_No hay salidas de consola guardadas (`evidence/consola_*.txt`)._\n"
    return "\n".join(code_block(path, path.stem.removeprefix("consola_").replace("_", " ")) for path in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", nargs="+",
                        help="marcas de tiempo de los benchmarks (por defecto, todos los de results/logs/)")
    args = parser.parse_args()
    stamps = find_stamps(args.stamp)
    if not stamps:
        sys.exit("No hay logs en results/logs/. Corre primero benchmark.py.")
    stamp_list = ", ".join(f"`{stamp}`" for stamp in stamps)

    speedup = read_text(RESULTS / "tabla_speedup.md").replace("# Tiempos por fase y speedup", "").strip()
    kernel = read_text(RESULTS / "kernel_benchmark.md").strip()

    document = f"""# Evidencia de ejecución – Proyecto 3 (ASL + CUDA)

Documento generado automáticamente por `scripts/armar_evidencia.py` el {datetime.now():%Y-%m-%d %H:%M}.
Ejecuciones del benchmark incluidas: {stamp_list} (cada una con su log, su CSV de CPU por núcleo y su CSV de
nvidia-smi con la misma marca de tiempo). Si hay más de una, es porque el benchmark oficial se completó en partes
sobre el mismo `results/tiempos.csv`, con el mismo hardware, datos e hiperparámetros.

Objetivo (rúbrica, 30 pts): demostrar con **logs** y **capturas de los monitores de recursos** que el entorno
reconoció y usó el hardware asignado **en cada fase**:

| Fase | Hardware asignado | Dónde se demuestra |
|---|---|---|
| FASE 1 · Preparación de imágenes (leer JPEG, decodificar, resize) | **CPU multinúcleo** | §2 (tabla por fase), capturas 03–04, figura de núcleos |
| FASE 2 · Entrenamiento y evaluación de la red | **GPU NVIDIA (CUDA)** | §2, §3 (`GPU detectada`), capturas 05–07 y 10–12, figura de GPU |
| Baseline secuencial | **CPU** (sin GPU) | §2 (GPU-Util ≈ 0 %), captura 09 |

---

## 1. Hardware y software detectados

{hardware_section()}
{code_block(EVIDENCE / "check_env.txt", "Salida de check_env.py")}
## 2. Uso de hardware por fase (a partir de los logs de monitoreo)

{phase_hardware_table(stamps)}
## 3. Logs del benchmark

{log_excerpt(stamps)}
## 4. Salidas de consola guardadas

{code_block(EVIDENCE / "nvidia_smi_entrenamiento.txt", "nvidia-smi durante el entrenamiento en GPU (evidencia_fases.py)")}
{console_section()}
## 5. Capturas de pantalla de los monitores de recursos

{screenshots_section()}
## 6. Figuras generadas a partir de los logs

{figures_section()}
## 7. Tiempos y speedup (resultado del benchmark)

{speedup or "_Falta `results/tabla_speedup.md`._"}

### Kernel CUDA propio (Numba) vs NumPy

{kernel or "_Falta `results/kernel_benchmark.md` (correr scripts/bench_kernel.py)._"}
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Documento de evidencia: {OUTPUT}")


if __name__ == "__main__":
    main()
