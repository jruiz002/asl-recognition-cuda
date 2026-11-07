"""VERSIÓN PARALELA: CPU multinúcleo para los datos + GPU (CUDA) para la red.

    ┌──────────────── FASE 1: PREPARACIÓN (CPU multinúcleo) ────────────────┐
    │ ProcessPoolExecutor(núcleos): JPEG → RGB → resize 64×64 → uint8 memmap │
    │ (+ con augmentation: DataLoader con N workers que aumentan cada época)  │
    └───────────────────────────────┬────────────────────────────────────────┘
                                    │ transferencia H2D en uint8 (pinned, non_blocking)
    ┌───────────────────────────────▼──── FASE 2: ENTRENAMIENTO (GPU) ──────┐
    │ kernel Numba uint8→float32 CHW · conv/BN/ReLU/pool/FC (cuDNN/cuBLAS)   │
    │ pérdida · backward · AdamW · evaluación                                │
    └────────────────────────────────────────────────────────────────────────┘

Se usa el MISMO modelo (``src.neural.build_model``), la misma inicialización
(``seed_everything``), la misma partición, el mismo orden de lotes, el mismo
optimizador y los mismos hiperparámetros que la versión secuencial.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from proyecto3.common.config import BenchmarkConfig
from proyecto3.common.timing import CudaEventAccumulator, PhaseTimer
from proyecto3.parallel.cpu_prep import default_workers, prepare_splits_parallel
from proyecto3.parallel.gpu_train import (
    configure_cuda,
    evaluate_resident,
    train_epoch_resident,
    train_epoch_streaming,
    to_device_uint8,
)
from src.neural import NeuralExperiment, SafeAugmentation, build_model, seed_everything


# ----------------- FASE 1 (CPU): dataset con augmentation por worker -----------
class AugmentedUint8Dataset(Dataset):
    """Igual que ``ASLImageDataset`` con caché + augmentation del lab, pero:

    - abre el memmap de forma diferida dentro de cada worker (no se copian 150 MB
      por pickle a cada proceso en Windows);
    - devuelve ``uint8 HWC``: la normalización /255 y el cambio a CHW los hace el
      kernel CUDA en la GPU (4× menos bytes por PCIe).
    """

    def __init__(self, cache_path: Path, labels: list[int], seed: int) -> None:
        self.cache_path = str(cache_path)
        self.labels = labels
        self.seed = seed
        self.cache = None
        self.augmentation = SafeAugmentation(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        from PIL import Image

        if self.cache is None:
            self.cache = np.load(self.cache_path, mmap_mode="r")
        image = self.augmentation(Image.fromarray(np.asarray(self.cache[index])))
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()), self.labels[index]


def _worker_init(worker_id: int) -> None:
    """Cada worker recibe su propio generador de augmentation (semilla + id).

    Sin esto, todos los workers tendrían una copia del mismo generador y producirían
    las mismas transformaciones para lotes distintos.
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.augmentation = SafeAugmentation(info.dataset.seed + 1000 * (worker_id + 1))


def default_loader_workers() -> int:
    """Workers del DataLoader: hasta 8 (en Windows cada uno es un proceso que importa torch)."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# --------------------------------- Pipeline ------------------------------------
def run_parallel(
    experiment: NeuralExperiment,
    manifests: dict[str, pd.DataFrame],
    class_names: list[str],
    config: BenchmarkConfig,
    timer: PhaseTimer,
    cache_dir: Path,
) -> dict:
    started_total = time.perf_counter()
    seed_everything(config.seed)
    torch.set_num_threads(config.threads or min(8, max(1, torch.get_num_threads())))

    # ===================== Configuración de CUDA (FASE 2) =======================
    device = configure_cuda()
    model = build_model(
        experiment.architecture, len(class_names), experiment.dropout, experiment.width
    ).to(device)

    label_lookup = {name: index for index, name in enumerate(class_names)}
    labels = {
        split: [label_lookup[label] for label in frame["label"]]
        for split, frame in manifests.items()
    }

    # =================== FASE 1: PREPARACIÓN (CPU multinúcleo) ===================
    workers = config.workers or default_workers()
    with timer.fase("preparacion_cpu"):
        caches = prepare_splits_parallel(
            {split: frame["path"].astype(str).tolist() for split, frame in manifests.items()},
            cache_dir,
            config.image_size,
            workers,
        )
        # Lectura a RAM de los conjuntos que irán completos a la GPU.
        resident_splits = ["validation", "test"] + ([] if experiment.augment else ["train"])
        host_arrays = {split: np.load(caches[split]) for split in resident_splits}
    print(f"  [paralelo] preparación con {workers} procesos de CPU", flush=True)

    # ============== Transferencia CPU → GPU (uint8, memoria fijada) =============
    with timer.fase("transferencia_h2d"):
        gpu_images = {split: to_device_uint8(host_arrays[split], device) for split in resident_splits}
        gpu_labels = {
            split: torch.tensor(labels[split], dtype=torch.int64).to(device, non_blocking=True)
            for split in manifests
        }

    train_loader = None
    if experiment.augment:
        loader_workers = default_loader_workers()
        train_loader = DataLoader(
            AugmentedUint8Dataset(caches["train"], labels["train"], config.seed),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=loader_workers,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=device.type == "cuda",
            worker_init_fn=_worker_init,
            generator=torch.Generator().manual_seed(config.seed),
        )
        print(f"  [paralelo] augmentation con {loader_workers} workers de CPU", flush=True)

    # ======================== FASE 2: ENTRENAMIENTO (GPU) ========================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(config.seed)   # mismo que make_loaders
    kernel_timer, h2d_timer = CudaEventAccumulator(), CudaEventAccumulator()
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        with timer.fase("entrenamiento"):
            if experiment.augment:
                stats = train_epoch_streaming(
                    model, optimizer, criterion, train_loader, device, kernel_timer, h2d_timer
                )
            else:
                stats = train_epoch_resident(
                    model, optimizer, criterion, gpu_images["train"], gpu_labels["train"],
                    config.batch_size, generator, kernel_timer,
                )
            train_loss = float(stats["loss_sum"]) / stats["count"]   # 1 sincronización/época
            train_accuracy = float(stats["correct"]) / stats["count"]

        with timer.fase("evaluacion"):
            validation = evaluate_resident(
                model, gpu_images["validation"], gpu_labels["validation"],
                config.batch_size, kernel_timer,
            )
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "train_accuracy": train_accuracy, **validation})
            if validation["macro_f1"] > best_f1 + 1e-4:
                best_f1 = validation["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        print(
            f"  [paralelo]   {experiment.name} epoch={epoch}/{config.epochs} "
            f"loss={train_loss:.4f} val_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    with timer.fase("evaluacion"):
        model.load_state_dict(best_state)
        test_metrics = evaluate_resident(
            model, gpu_images["test"], gpu_labels["test"], config.batch_size, kernel_timer
        )

    timer.add("kernel_normalizacion", kernel_timer.total_seconds())
    timer.add("h2d_en_bucle", h2d_timer.total_seconds())
    timer.add("total", time.perf_counter() - started_total)

    # Cierre de los workers persistentes, FUERA del tiempo total: es limpieza de
    # procesos del sistema operativo, no preparación de imágenes ni cómputo de la red.
    # (En macOS con PyTorch 2.15 cada worker tarda ~5 s en cerrarse; se registra en
    # el log para que quede documentado.)
    if train_loader is not None:
        closing = time.perf_counter()
        del train_loader
        print(f"  [paralelo] cierre de workers (no contabilizado): "
              f"{time.perf_counter() - closing:.2f} s", flush=True)
    return {
        "device": str(device),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "val_macro_f1": best_f1,
        "epochs_ran": len(history),
        "final_train_loss": history[-1]["train_loss"],
    }
