"""Modo EVIDENCIA: mantiene cada fase activa el tiempo suficiente para las capturas.

En el benchmark real, la preparación paralela dura pocos segundos y es difícil
capturarla. Este script repite cada fase durante ``--seconds`` segundos y avisa en
pantalla QUÉ capturar y CUÁNDO:

    1a. Preparación SECUENCIAL (lab original)  → Administrador de tareas: 1 núcleo ocupado
    1b. Preparación PARALELA (todos los núcleos) → Administrador de tareas: todos los núcleos
    2.  Entrenamiento en GPU (CUDA)             → Administrador de tareas (GPU) + nvidia-smi
        (además guarda automáticamente evidence/nvidia_smi_entrenamiento.txt)

Uso:  python scripts/evidencia_fases.py --seconds 40
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def banner(text: str) -> None:
    print("\n" + "#" * 78 + f"\n# {text}\n" + "#" * 78, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=int, default=40, help="duración de cada fase")
    parser.add_argument("--skip-sequential", action="store_true")
    args = parser.parse_args()
    from proyecto3.common import consola_utf8

    consola_utf8()

    import pandas as pd
    import torch
    from torch import nn

    from proyecto3.common.paths import load_portable_manifest, split_manifests
    from proyecto3.parallel.cpu_prep import default_workers, prepare_splits_parallel
    from proyecto3.parallel.gpu_kernels import normalize_torch_batch
    from proyecto3.parallel.gpu_train import configure_cuda, to_device_uint8
    from src.neural import build_image_cache, build_model

    data = load_portable_manifest(ROOT / "data" / "manifests" / "all.csv", ROOT / "data" / "raw")
    manifests, class_names = split_manifests(data)
    paths = manifests["train"]["path"].astype(str).tolist()
    work = ROOT / "data" / "bench_cache" / "evidencia"
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)

    if not args.skip_sequential:
        banner("FASE 1a – PREPARACIÓN SECUENCIAL (lab original, 1 proceso). "
               "CAPTURA AHORA: Administrador de tareas → Rendimiento → CPU "
               "(clic derecho → Cambiar gráfico a → Procesadores lógicos)")
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            shutil.rmtree(work, ignore_errors=True)
            build_image_cache(pd.DataFrame({"path": paths[:3000]}), work / "seq.npy")
        print("Fin de la fase 1a.", flush=True)
        time.sleep(3)

    banner(f"FASE 1b – PREPARACIÓN PARALELA ({default_workers()} procesos). "
           "CAPTURA AHORA: Administrador de tareas → CPU con todos los núcleos")
    deadline = time.perf_counter() + args.seconds
    rounds = 0
    while time.perf_counter() < deadline:
        shutil.rmtree(work, ignore_errors=True)
        caches = prepare_splits_parallel({"train": paths}, work, 64)
        rounds += 1
    print(f"Fin de la fase 1b ({rounds} pasadas completas por {len(paths):,} imágenes).", flush=True)
    time.sleep(3)

    banner("FASE 2 – ENTRENAMIENTO EN GPU (CUDA). CAPTURA AHORA: Administrador de tareas → GPU "
           "(cambiar un gráfico a 'Cuda' o 'Compute_0') y la terminal con nvidia-smi")
    import numpy as np

    device = configure_cuda()
    images = to_device_uint8(np.load(caches["train"]), device)
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = torch.tensor([lookup[label] for label in manifests["train"]["label"]], device=device)
    model = build_model("cnn_regularized", len(class_names), 0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    deadline = time.perf_counter() + args.seconds
    snapshot_taken = False
    epochs = 0
    while time.perf_counter() < deadline:
        order = torch.randperm(images.shape[0], device=device)
        for start in range(0, order.numel(), 64):
            index = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(normalize_torch_batch(images[index])), labels[index])
            loss.backward()
            optimizer.step()
        epochs += 1
        if not snapshot_taken and shutil.which("nvidia-smi"):
            snapshot = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
            (evidence / "nvidia_smi_entrenamiento.txt").write_text(snapshot, encoding="utf-8")
            print(snapshot, flush=True)
            snapshot_taken = True
        print(f"  época {epochs} · loss {loss.item():.4f}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    banner("FIN DEL MODO EVIDENCIA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
