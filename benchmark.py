"""Benchmark del Proyecto 3: versión SECUENCIAL (lab original) vs PARALELA (CPU multinúcleo + CUDA).

Ejecuta ambas versiones con los mismos datos y las mismas condiciones, N veces cada
una (más un calentamiento que no se cuenta). Mide cada fase y guarda:

    results/tiempos.csv           una fila por corrida (segundos por fase + métricas)
    results/resumen.csv           media y desviación estándar por versión/experimento/fase
    results/tabla_speedup.md      tabla lista para pegar en el informe
    results/hw_info.json          hardware y versiones de software
    results/logs/benchmark_*.log  salida completa de consola
    evidence/nvidia_smi_*.csv     uso de la GPU cada segundo (si hay nvidia-smi)
    evidence/cpu_por_nucleo_*.csv uso de cada núcleo del CPU, etiquetado por fase
    results/figuras/*.png         gráficas (scripts/make_plots.py)

Uso típico:
    python benchmark.py --fresh                         # benchmark completo (laptop)
    python benchmark.py --sample 20 --epochs 1 --runs 1 # prueba rápida (Mac)
    python benchmark.py --only parallel --runs 3        # solo una versión
"""

from __future__ import annotations

# IMPORTANTE: este archivo mantiene ligeros los imports de nivel superior. En
# Windows, los procesos de la FASE 1 se crean con "spawn" y reimportan el módulo
# principal; torch y pandas se importan dentro de main() para no cargarlos en
# cada proceso de trabajo.
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


class _Tee:
    """Duplica la salida de consola en el archivo de log."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("sequential", "parallel", "both"), default="both")
    parser.add_argument("--runs", type=int, default=5, help="repeticiones medidas (por defecto 5)")
    parser.add_argument("--epochs", type=int, default=6, help="épocas por corrida (lab: 6)")
    parser.add_argument("--sample", type=int, default=None,
                        help="imágenes por clase y partición (modo prueba rápida)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="nombres de entrenar_modelos.NEURAL_EXPERIMENTS")
    parser.add_argument("--threads", type=int, default=None,
                        help="hilos intra-op de PyTorch (por defecto, igual que el lab)")
    parser.add_argument("--workers", type=int, default=None,
                        help="procesos de CPU para la FASE 1 paralela (por defecto, todos)")
    parser.add_argument("--no-warmup", action="store_true", help="omite la corrida de calentamiento")
    parser.add_argument("--data-root", type=Path, default=None, help="carpeta del dataset (data/raw)")
    parser.add_argument("--fresh", action="store_true",
                        help="borra results/tiempos.csv antes de empezar")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def main() -> int:
    args = parse_args()

    from proyecto3.common import consola_utf8

    consola_utf8()
    import gc

    import pandas as pd

    from proyecto3.common.config import BenchmarkConfig, DEFAULT_EXPERIMENTS, ROOT
    from proyecto3.common.hw_info import collect_hw_info
    from proyecto3.common.monitors import CpuPerCoreMonitor, NvidiaSmiLogger
    from proyecto3.common.paths import load_portable_manifest, split_manifests, stratified_sample
    from proyecto3.common.timing import PHASES, SUBMETRICS, PhaseTimer
    from proyecto3.parallel.pipeline import run_parallel
    from proyecto3.sequential.pipeline import run_sequential

    config = BenchmarkConfig(
        experiments=tuple(args.experiments or DEFAULT_EXPERIMENTS),
        epochs=args.epochs,
        runs=args.runs,
        warmup=not args.no_warmup,
        sample_per_class=args.sample,
        threads=args.threads,
        workers=args.workers,
    )
    if args.data_root:
        config.data_root = args.data_root

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config.results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config.evidence_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"benchmark_{stamp}.log").open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    banner(f"BENCHMARK PROYECTO 3 – {stamp}")
    hw = collect_hw_info()
    hw["argumentos"] = {key: str(value) for key, value in vars(args).items()}
    (config.results_dir / "hw_info.json").write_text(json.dumps(hw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hw, indent=2, ensure_ascii=False))
    if not hw["cuda_disponible"]:
        print("\n*** AVISO: no se detectó GPU CUDA. La versión paralela usará CPU en la FASE 2. ***")
        print("*** Los tiempos de esta ejecución NO son válidos para el informe.            ***")

    # ----------------------- Datos: misma partición del lab -----------------------
    from entrenar_modelos import NEURAL_EXPERIMENTS

    catalog = {experiment.name: experiment for experiment in NEURAL_EXPERIMENTS}
    unknown = set(config.experiments) - set(catalog)
    if unknown:
        raise SystemExit(f"Experimentos desconocidos: {sorted(unknown)}")
    data = load_portable_manifest(config.manifest, config.data_root)
    data = stratified_sample(data, config.sample_per_class, config.seed)
    manifests, class_names = split_manifests(data)
    print(f"\nImágenes: { {split: len(frame) for split, frame in manifests.items()} } "
          f"· clases: {len(class_names)} · épocas: {config.epochs} · corridas: {config.runs}")

    # ------------------------------ Monitores -------------------------------------
    gpu_monitor = NvidiaSmiLogger(config.evidence_dir / f"nvidia_smi_{stamp}.csv")
    gpu_monitor.start()
    cpu_monitor = CpuPerCoreMonitor(config.evidence_dir / f"cpu_por_nucleo_{stamp}.csv")
    cpu_monitor.start()

    shutil.rmtree(config.work_dir, ignore_errors=True)   # restos de ejecuciones anteriores
    csv_path = config.results_dir / "tiempos.csv"
    if args.fresh and csv_path.exists():
        csv_path.unlink()
    # Con --threads la versión se registra con otro nombre para no mezclarla con el
    # baseline del lab en los promedios (p. ej. "secuencial_1hilos").
    suffix = f"_{config.threads}hilos" if config.threads else ""
    versions = {
        "sequential": (f"secuencial{suffix}", run_sequential),
        "parallel": (f"paralela{suffix}", run_parallel),
    }
    selected = list(versions) if args.only == "both" else [args.only]

    try:
        for experiment_name in config.experiments:
            experiment = catalog[experiment_name]
            for version_key in selected:
                version_name, runner = versions[version_key]
                run_ids = ([0] if config.warmup else []) + list(range(1, config.runs + 1))
                for run_id in run_ids:
                    is_warmup = run_id == 0
                    # El calentamiento usa 1 época: basta para crear el contexto CUDA,
                    # compilar el kernel Numba (JIT), elegir algoritmos de cuDNN y
                    # llevar los JPEG a la caché de archivos del sistema operativo.
                    run_config = config
                    if is_warmup:
                        from dataclasses import replace
                        run_config = replace(config, epochs=1)
                    label = "calentamiento" if is_warmup else f"corrida {run_id}/{config.runs}"
                    banner(f"{version_name.upper()} · {experiment_name} · {label}")
                    cpu_monitor.set_context(version=version_name, experimento=experiment_name,
                                            corrida="warmup" if is_warmup else run_id)
                    timer = PhaseTimer(on_phase=cpu_monitor.set_phase)
                    cache_dir = config.work_dir / f"{version_key}_{experiment_name}_{run_id}"
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    try:
                        metrics = runner(experiment, manifests, class_names, run_config, timer, cache_dir)
                    finally:
                        # En Windows un memmap abierto impide borrar el archivo: primero se
                        # liberan los objetos de la corrida (DataLoaders, datasets, memmaps).
                        gc.collect()
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    seconds = timer.as_dict()
                    print("  tiempos (s): " + " · ".join(f"{name}={seconds[name]:.2f}" for name in PHASES))
                    if is_warmup:
                        continue
                    row = {
                        "fecha": datetime.now().isoformat(timespec="seconds"),
                        "version": version_name,
                        "experimento": experiment_name,
                        "corrida": run_id,
                        "epocas": config.epochs,
                        "imagenes_train": len(manifests["train"]),
                        "muestra_por_clase": config.sample_per_class or "completo",
                        **{f"t_{name}": seconds[name] for name in (*PHASES, *SUBMETRICS)},
                        **metrics,
                        "cpu": hw["cpu"],
                        "gpu": hw.get("gpu", "-"),
                    }
                    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    finally:
        cpu_monitor.stop()
        gpu_monitor.stop()
        gc.collect()
        shutil.rmtree(config.work_dir, ignore_errors=True)

    banner("RESUMEN")
    from scripts.make_plots import summarize

    summarize(csv_path, config.results_dir)
    if not args.no_plots:
        from scripts.make_plots import make_all_plots

        make_all_plots(csv_path, config.results_dir / "figuras", config.evidence_dir)
    print(f"\nLog completo: {log_dir / f'benchmark_{stamp}.log'}")
    sys.stdout = sys.__stdout__
    log_file.close()
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    exit_code = main()
    print(f"Duración total del benchmark: {(time.perf_counter() - started) / 60:.1f} min")
    raise SystemExit(exit_code)
