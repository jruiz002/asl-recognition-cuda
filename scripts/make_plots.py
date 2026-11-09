"""Tablas y gráficas del Proyecto 3 a partir de results/tiempos.csv y evidence/*.csv.

Genera:
    results/resumen.csv                 media, desviación estándar, n por versión/experimento/fase
    results/tabla_speedup.md            tabla markdown (tiempos ± sd y speedup por fase)
    results/figuras/01_tiempos_por_fase.png
    results/figuras/02_speedup_por_fase.png
    results/figuras/03_cpu_nucleos_<version>.png   mapa de calor de uso por núcleo + fases
    results/figuras/03_cpu_preparacion_por_nucleo.png  uso medio por núcleo en la FASE 1 (sec vs par)
    results/figuras/04_gpu_uso.png                 utilización y memoria de la GPU (nvidia-smi)

Uso:  python scripts/make_plots.py   (o se llama automáticamente desde benchmark.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASES = ("preparacion_cpu", "transferencia_h2d", "entrenamiento", "evaluacion", "total")
PHASE_LABELS = {
    "preparacion_cpu": "Preparación (CPU)",
    "transferencia_h2d": "Transferencia H2D",
    "entrenamiento": "Entrenamiento",
    "evaluacion": "Evaluación",
    "total": "Total",
    "carga_lotes_cpu": "Carga de lotes (CPU)",
    "kernel_normalizacion": "Kernel normalización",
    "h2d_en_bucle": "H2D en el bucle",
}
# Paleta categórica de referencia (orden fijo): 1 = azul, 2 = naranja.
COLORS = {"secuencial": "#2a78d6", "paralela": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=INK_2)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


# ------------------------------- Tablas ---------------------------------------
def summarize(csv_path: Path, output_dir: Path) -> pd.DataFrame | None:
    """Media ± desviación por fase y speedup = t_secuencial / t_paralela."""
    if not csv_path.exists():
        print(f"No existe {csv_path}")
        return None
    data = pd.read_csv(csv_path)
    time_columns = [column for column in data.columns if column.startswith("t_")]
    long = data.melt(
        id_vars=["version", "experimento", "corrida"],
        value_vars=time_columns, var_name="fase", value_name="segundos",
    )
    long["fase"] = long["fase"].str.removeprefix("t_")
    summary = (
        long.groupby(["experimento", "fase", "version"])["segundos"]
        .agg(media="mean", desviacion="std", n="count").reset_index()
    )
    summary.to_csv(output_dir / "resumen.csv", index=False)

    lines = ["# Tiempos por fase y speedup", "",
             f"Fuente: `{csv_path.name}` · media ± desviación estándar en segundos · "
             "speedup = t_secuencial / t_paralela", ""]
    metric_cols = ["test_accuracy", "test_macro_f1"]
    for experiment, frame in summary.groupby("experimento", sort=False):
        lines += [f"## {experiment}", "",
                  "| Fase | Secuencial (s) | Paralela (s) | Speedup |",
                  "|---|---:|---:|---:|"]
        pivot = frame.pivot(index="fase", columns="version", values=["media", "desviacion", "n"])
        for phase in [*PHASES, "carga_lotes_cpu", "kernel_normalizacion", "h2d_en_bucle"]:
            if phase not in pivot.index:
                continue
            cells = []
            for version in ("secuencial", "paralela"):
                if ("media", version) in pivot.columns and not np.isnan(pivot.loc[phase, ("media", version)]):
                    mean = pivot.loc[phase, ("media", version)]
                    std = pivot.loc[phase, ("desviacion", version)]
                    cells.append(f"{mean:.2f} ± {0 if np.isnan(std) else std:.2f}")
                else:
                    cells.append("–")
            speedup = "–"
            if ("media", "secuencial") in pivot.columns and ("media", "paralela") in pivot.columns:
                seq, par = pivot.loc[phase, ("media", "secuencial")], pivot.loc[phase, ("media", "paralela")]
                if seq > 0 and par > 0:
                    speedup = f"**{seq / par:.2f}×**" if phase == "total" else f"{seq / par:.2f}×"
            name = PHASE_LABELS.get(phase, phase)
            if phase not in PHASES:
                name = f"↳ {name} (incluida en entrenamiento)"
            lines.append(f"| {name} | {cells[0]} | {cells[1]} | {speedup} |")
        quality = data[data["experimento"] == experiment].groupby("version")[metric_cols].mean()
        lines += ["", "Calidad del modelo (promedio de las corridas, conjunto de prueba):", "",
                  "| Versión | Accuracy | Macro F1 |", "|---|---:|---:|"]
        for version, row in quality.iterrows():
            lines.append(f"| {version} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} |")
        lines.append("")
    table = "\n".join(lines)
    (output_dir / "tabla_speedup.md").write_text(table, encoding="utf-8")
    print(table)
    return summary


# ------------------------------- Gráficas -------------------------------------
def plot_phase_times(data: pd.DataFrame, figure_dir: Path) -> None:
    experiments = list(dict.fromkeys(data["experimento"]))
    figure, axes = plt.subplots(1, len(experiments), figsize=(6.5 * len(experiments), 4.8), squeeze=False)
    for axis, experiment in zip(axes[0], experiments):
        frame = data[data["experimento"] == experiment]
        positions = np.arange(len(PHASES))
        versions = [version for version in ("secuencial", "paralela") if version in set(frame["version"])]
        width = 0.8 / max(1, len(versions))
        for offset, version in enumerate(versions):
            subset = frame[frame["version"] == version]
            means = [subset[f"t_{phase}"].mean() for phase in PHASES]
            stds = [subset[f"t_{phase}"].std(ddof=1) if len(subset) > 1 else 0 for phase in PHASES]
            x = positions + (offset - (len(versions) - 1) / 2) * width
            axis.bar(x, means, width * 0.92, yerr=stds, color=COLORS[version], label=version,
                     error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
            for xi, value in zip(x, means):
                axis.text(xi, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        axis.set_xticks(positions, [PHASE_LABELS[phase] for phase in PHASES], rotation=20, ha="right")
        axis.set_ylabel("Segundos (media ± sd)", color=INK_2)
        axis.set_title(experiment, color=INK)
        _style(axis)
        axis.legend(frameon=False)
    figure.suptitle("Tiempo por fase: secuencial vs paralela", color=INK)
    figure.tight_layout()
    figure.savefig(figure_dir / "01_tiempos_por_fase.png", dpi=180)
    plt.close(figure)


def plot_speedup(data: pd.DataFrame, figure_dir: Path) -> None:
    if not {"secuencial", "paralela"} <= set(data["version"]):
        return
    phases = [phase for phase in PHASES if phase != "transferencia_h2d"]   # no existe en secuencial
    experiments = list(dict.fromkeys(data["experimento"]))
    figure, axes = plt.subplots(1, len(experiments), figsize=(5.5 * len(experiments), 4.5), squeeze=False)
    for axis, experiment in zip(axes[0], experiments):
        frame = data[data["experimento"] == experiment]
        seq = frame[frame["version"] == "secuencial"]
        par = frame[frame["version"] == "paralela"]
        values = [seq[f"t_{phase}"].mean() / par[f"t_{phase}"].mean() for phase in phases]
        colors = [COLORS["paralela"] if phase == "total" else "#f3a37f" for phase in phases]
        bars = axis.bar([PHASE_LABELS[phase] for phase in phases], values, color=colors, width=0.6)
        axis.axhline(1.0, color=INK_2, linewidth=1, linestyle="--", label="1× = sin mejora")
        axis.legend(frameon=False, loc="upper right", fontsize=8)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}×",
                      ha="center", va="bottom", fontsize=9, color=INK)
        axis.set_ylabel("Speedup (t_sec / t_par)", color=INK_2)
        axis.set_title(experiment, color=INK)
        axis.tick_params(axis="x", rotation=20)
        _style(axis)
    figure.suptitle("Speedup por fase", color=INK)
    figure.tight_layout()
    figure.savefig(figure_dir / "02_speedup_por_fase.png", dpi=180)
    plt.close(figure)


def _load_cpu_samples(evidence_dir: Path) -> pd.DataFrame | None:
    """Junta TODOS los ``cpu_por_nucleo_*.csv`` (el benchmark puede quedar repartido en varias ejecuciones)."""
    files = sorted(evidence_dir.glob("cpu_por_nucleo_*.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(path, dtype={"corrida": str}) for path in files], ignore_index=True)


def plot_cpu_cores(evidence_dir: Path, figure_dir: Path) -> None:
    """Mapa de calor núcleo × tiempo de la PRIMERA corrida medida de cada versión."""
    samples = _load_cpu_samples(evidence_dir)
    if samples is None:
        return
    core_columns = [column for column in samples.columns if column.startswith("nucleo_")]
    for version in ("secuencial", "paralela"):
        frame = samples[(samples["version"] == version) & (samples["corrida"].astype(str) == "1")]
        if frame.empty:
            continue
        experiment = frame["experimento"].iloc[0]
        frame = frame[frame["experimento"] == experiment]
        times = frame["t_rel_s"].to_numpy() - frame["t_rel_s"].iloc[0]
        matrix = frame[core_columns].to_numpy().T
        figure, (ax_phase, ax_heat) = plt.subplots(
            2, 1, figsize=(11, 1.2 + 0.22 * len(core_columns)), sharex=True,
            gridspec_kw={"height_ratios": [1, max(3, len(core_columns) // 3)]},
        )
        # Franja superior: fase activa.
        phase_colors = {"preparacion_cpu": "#2a78d6", "transferencia_h2d": "#eda100",
                        "entrenamiento": "#1baf7a", "evaluacion": "#6250d6", "idle": "#e4e3df"}
        phases = frame["fase"].to_numpy()
        for index in range(len(times)):
            end = times[index + 1] if index + 1 < len(times) else times[index] + 0.5
            ax_phase.axvspan(times[index], end, color=phase_colors.get(phases[index], "#e4e3df"), linewidth=0)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for name, color in phase_colors.items()
                   if name in set(phases)]
        labels = [PHASE_LABELS.get(name, "Otra / inactivo") for name in phase_colors if name in set(phases)]
        ax_phase.legend(handles, labels, ncol=len(labels), loc="lower left", bbox_to_anchor=(0, 1.05),
                        frameon=False, fontsize=8)
        ax_phase.set_yticks([])
        ax_phase.set_ylabel("Fase", rotation=0, ha="right", va="center", color=INK_2)
        image = ax_heat.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100,
                               extent=[times[0], times[-1] if len(times) > 1 else 1, len(core_columns) - 0.5, -0.5],
                               interpolation="nearest")
        ax_heat.set_ylabel("Núcleo lógico", color=INK_2)
        ax_heat.set_xlabel("Segundos desde el inicio de la corrida", color=INK_2)
        figure.colorbar(image, ax=[ax_phase, ax_heat], label="% uso", fraction=0.03, pad=0.01)
        figure.suptitle(f"Uso por núcleo del CPU – versión {version} ({experiment}, corrida 1)", color=INK, y=1.02)
        figure.savefig(figure_dir / f"03_cpu_nucleos_{version}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def plot_prep_cores(evidence_dir: Path, figure_dir: Path) -> None:
    """Uso medio de cada núcleo DURANTE la preparación: secuencial (1 proceso) vs paralela.

    Es la evidencia directa de la FASE 1: en la secuencial trabaja ~1 núcleo; en la
    paralela, todos. Promedia todas las muestras de la fase en todas las corridas medidas.
    """
    samples = _load_cpu_samples(evidence_dir)
    if samples is None:
        return
    samples = samples[(samples["fase"] == "preparacion_cpu") & (samples["corrida"].astype(str) != "warmup")]
    versions = [version for version in ("secuencial", "paralela") if version in set(samples["version"])]
    if not versions:
        return
    core_columns = [column for column in samples.columns if column.startswith("nucleo_")]
    positions = np.arange(len(core_columns))
    width = 0.8 / len(versions)
    figure, axis = plt.subplots(figsize=(max(8, 0.45 * len(core_columns) + 3), 4.2))
    for offset, version in enumerate(versions):
        frame = samples[samples["version"] == version]
        means = frame[core_columns].mean().to_numpy()
        x = positions + (offset - (len(versions) - 1) / 2) * width
        axis.bar(x, means, width * 0.92, color=COLORS[version],
                 label=f"{version} (media {frame['cpu_total'].mean():.0f} % · {len(frame)} muestras)")
    axis.set_xticks(positions, [column.split("_")[1] for column in core_columns])
    axis.set_xlabel("Núcleo lógico", color=INK_2)
    axis.set_ylabel("Uso medio durante la preparación (%)", color=INK_2)
    axis.set_ylim(0, 105)
    axis.set_title("FASE 1 · Preparación de imágenes: uso de cada núcleo del CPU", color=INK)
    _style(axis)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(figure_dir / "03_cpu_preparacion_por_nucleo.png", dpi=180)
    plt.close(figure)


def plot_gpu(evidence_dir: Path, figure_dir: Path) -> None:
    """Uso de la GPU en TODAS las ejecuciones del benchmark, una tras otra, con cada tramo rotulado."""
    frames, segments = [], []
    for path in sorted(evidence_dir.glob("nvidia_smi_2*.csv")):
        if path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [column.split(" [")[0].strip() for column in frame.columns]
        frame = frame[frame["timestamp"].astype(str) != "timestamp"]
        if frame.empty:
            continue
        # Rótulo del tramo: versiones y modelos medidos en esa ejecución (del CSV de CPU con la misma marca).
        label = path.stem.removeprefix("nvidia_smi_")
        # Solo las combinaciones con al menos una corrida TERMINADA (línea "tiempos (s)" en el log), para no
        # rotular intentos interrumpidos.
        log = evidence_dir.parent / "results" / "logs" / f"benchmark_{label}.log"
        if log.exists():
            done, current = [], None
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                header = line.strip()
                if " · " in header and header.split(" · ")[0] in ("SECUENCIAL", "PARALELA"):
                    parts = header.split(" · ")
                    # El calentamiento no cuenta: solo las corridas medidas ("corrida N/M").
                    current = f"{parts[0].lower()} · {parts[1]}" if "corrida" in header else None
                elif "tiempos (s)" in line and current and current not in done:
                    done.append(current)
            label = "\n".join(done) or label
        start = sum(len(f) for f in frames)
        segments.append((start, start + len(frame), label))
        frames.append(frame)
    if not frames:
        return
    smi = pd.concat(frames, ignore_index=True)
    numeric = lambda column: pd.to_numeric(smi[column].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    seconds = np.arange(len(smi))
    figure, (ax_util, ax_mem) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for start, end, label in segments:
        for axis in (ax_util, ax_mem):
            axis.axvline(start, color=INK_2, linestyle=":", linewidth=1)
        ax_util.text((start + end) / 2, 103, label, ha="center", va="bottom", fontsize=7, color=INK_2)
    ax_util.plot(seconds, numeric("utilization.gpu"), color=COLORS["paralela"], linewidth=2)
    ax_util.set_ylabel("Uso de núcleos GPU (%)", color=INK_2)
    ax_util.set_ylim(0, 105)
    ax_mem.plot(seconds, numeric("memory.used"), color=COLORS["secuencial"], linewidth=2)
    total = numeric("memory.total").max()
    ax_mem.axhline(total, color=INK_2, linestyle="--", linewidth=1)
    ax_mem.text(seconds[-1] if len(seconds) else 0, total, " memoria total", va="bottom", ha="right", fontsize=8, color=INK_2)
    ax_mem.set_ylabel("Memoria usada (MiB)", color=INK_2)
    ax_mem.set_xlabel("Segundos de monitoreo (muestreo 1 s; ejecuciones del benchmark una tras otra)", color=INK_2)
    for axis in (ax_util, ax_mem):
        _style(axis)
    figure.suptitle(f"GPU durante el benchmark – {smi['name'].iloc[0] if 'name' in smi else ''}", color=INK)
    figure.tight_layout()
    figure.savefig(figure_dir / "04_gpu_uso.png", dpi=180)
    plt.close(figure)


def make_all_plots(csv_path: Path, figure_dir: Path, evidence_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        data = pd.read_csv(csv_path)
        plot_phase_times(data, figure_dir)
        plot_speedup(data, figure_dir)
    plot_cpu_cores(evidence_dir, figure_dir)
    plot_prep_cores(evidence_dir, figure_dir)
    plot_gpu(evidence_dir, figure_dir)
    print(f"Gráficas en {figure_dir}")


if __name__ == "__main__":
    results = ROOT / "results"
    summarize(results / "tiempos.csv", results)
    make_all_plots(results / "tiempos.csv", results / "figuras", ROOT / "evidence")
