"""Speedup propio del kernel Numba de normalización (uint8 HWC → float32 CHW / 255).

Compara, para el conjunto de entrenamiento completo (12 180 × 64 × 64 × 3):

    A. CPU NumPy                 normalizar en el CPU (lo que hacía el lab)
    B. CPU NumPy + H2D float32   alternativa sin kernel: normalizar en CPU y copiar 4 B/valor
    C. H2D uint8 + kernel Numba  lo que hace la versión paralela: copiar 1 B/valor y normalizar en GPU
    D. Kernel Numba (solo cómputo) datos ya residentes en la GPU
    E. PyTorch nativo en GPU     x.float().div(255).permute(...) como referencia

Tiempos con CUDA events (GPU) y perf_counter (CPU); warm-up excluido; N repeticiones.
Salidas: results/kernel_benchmark.csv, results/kernel_benchmark.md, results/figuras/05_kernel.png

Uso:
    python scripts/bench_kernel.py --reps 20
    NUMBA_ENABLE_CUDASIM=1 python scripts/bench_kernel.py --images 32 --reps 2   # Mac (simulador)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proyecto3.parallel.gpu_kernels import (  # noqa: E402
    THREADS_PER_BLOCK,
    blocks_per_grid,
    launch_normalize,
    normalize_numpy,
    numba_cuda_available,
)


def time_cpu(function, reps: int) -> list[float]:
    function()   # calentamiento
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return samples


def time_gpu(function, reps: int) -> list[float]:
    """Mide con eventos CUDA de PyTorch (requiere GPU real)."""
    import torch

    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=12_180, help="imágenes (por defecto, el train completo)")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    from proyecto3.common import consola_utf8

    consola_utf8()

    import pandas as pd

    data = np.random.default_rng(42).integers(0, 256, (args.images, args.size, args.size, 3), dtype=np.uint8)
    rows: list[dict] = []

    def record(name: str, samples: list[float]) -> None:
        rows.append({"variante": name, "media_ms": 1000 * np.mean(samples),
                     "sd_ms": 1000 * np.std(samples, ddof=1) if len(samples) > 1 else 0.0,
                     "reps": len(samples)})
        print(f"{name:<38} {rows[-1]['media_ms']:10.3f} ms ± {rows[-1]['sd_ms']:.3f}")

    print(f"Datos: {data.shape} uint8 = {data.nbytes / 1e6:.1f} MB · "
          f"bloque {THREADS_PER_BLOCK} · grid {blocks_per_grid(args.images, args.size, args.size)}")
    record("A. CPU NumPy", time_cpu(lambda: normalize_numpy(data), args.reps))

    try:
        import torch
        real_gpu = torch.cuda.is_available()
    except ImportError:
        real_gpu = False

    if real_gpu:
        from numba import cuda

        from proyecto3.parallel.gpu_kernels import normalize_torch_batch

        host_u8 = torch.from_numpy(data).pin_memory()
        device_u8 = host_u8.cuda()
        output = torch.empty((args.images, 3, args.size, args.size), dtype=torch.float32, device="cuda")
        stream = cuda.external_stream(torch.cuda.current_stream().cuda_stream)
        src, dst = cuda.as_cuda_array(device_u8), cuda.as_cuda_array(output)

        def cpu_then_h2d():
            torch.from_numpy(normalize_numpy(data)).pin_memory().to("cuda", non_blocking=True)

        record("B. CPU NumPy + H2D float32", time_gpu(cpu_then_h2d, args.reps))
        record("C. H2D uint8 + kernel Numba", time_gpu(
            lambda: normalize_torch_batch(host_u8.to("cuda", non_blocking=True)), args.reps))
        record("D. Kernel Numba (solo cómputo)", time_gpu(lambda: launch_normalize(src, dst, stream), args.reps))
        record("E. PyTorch nativo en GPU", time_gpu(
            lambda: device_u8.float().div_(255.0).permute(0, 3, 1, 2).contiguous(), args.reps))
        reference = normalize_numpy(data[:256])
        np.testing.assert_array_equal(normalize_torch_batch(device_u8[:256]).cpu().numpy(), reference)
        print("Verificación: kernel == NumPy (bit a bit) ✔")
    elif numba_cuda_available():
        from numba import cuda

        print("Simulador CUDA (NUMBA_ENABLE_CUDASIM=1): solo valida corrección; los tiempos NO son válidos.")
        src = cuda.to_device(data)
        dst = cuda.device_array((args.images, 3, args.size, args.size), dtype=np.float32)
        record("D. Kernel Numba (SIMULADOR)", time_cpu(lambda: launch_normalize(src, dst), args.reps))
        np.testing.assert_array_equal(dst.copy_to_host(), normalize_numpy(data))
        print("Verificación: kernel == NumPy (bit a bit) ✔")
    else:
        print("Sin GPU ni simulador: solo se mide la variante CPU.")

    results = pd.DataFrame(rows)
    cpu_ms = results.loc[results["variante"] == "A. CPU NumPy", "media_ms"].iloc[0]
    results["speedup_vs_cpu"] = cpu_ms / results["media_ms"]
    out_dir = ROOT / "results"
    (out_dir / "figuras").mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "kernel_benchmark.csv", index=False)
    lines = [f"# Kernel de normalización: {args.images} imágenes {args.size}×{args.size}, {args.reps} repeticiones", "",
             "| Variante | Media (ms) | SD (ms) | Speedup vs CPU |", "|---|---:|---:|---:|"]
    lines += [f"| {r.variante} | {r.media_ms:.3f} | {r.sd_ms:.3f} | {r.speedup_vs_cpu:.2f}× |" for r in results.itertuples()]
    (out_dir / "kernel_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 0.6 + 0.6 * len(results)))
    colors = ["#2a78d6" if name.startswith(("A", "B")) else "#eb6834" for name in results["variante"]]
    axis.barh(results["variante"][::-1], results["media_ms"][::-1], color=colors[::-1], height=0.6,
              xerr=results["sd_ms"][::-1], error_kw={"ecolor": "#52514e", "capsize": 3})
    for index, (value, speed) in enumerate(zip(results["media_ms"][::-1], results["speedup_vs_cpu"][::-1])):
        axis.text(value, index, f"  {value:.2f} ms ({speed:.1f}×)", va="center", fontsize=9, color="#0b0b0b")
    axis.set_xlabel("Milisegundos (media ± sd) · azul = CPU, naranja = GPU", color="#52514e")
    axis.set_title("Normalización uint8 → float32 CHW: CPU vs kernel CUDA", color="#0b0b0b")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_dir / "figuras" / "05_kernel.png", dpi=180)
    print(f"\nResultados en {out_dir / 'kernel_benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
