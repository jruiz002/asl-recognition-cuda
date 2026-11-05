"""VERSIÓN SECUENCIAL (baseline) = pipeline ORIGINAL del Laboratorio 3, instrumentado.

Este módulo NO reimplementa nada: llama a las mismas funciones de ``src/neural.py``
(``seed_everything``, ``build_model``, ``make_loaders`` → ``build_image_cache`` +
``ASLImageDataset``, ``evaluate_loader``) y reproduce línea por línea el cuerpo de
``train_experiment``. Lo único que se agregó son los timers por fase.

Diferencias respecto a ``train_experiment`` (solo de medición, documentadas en
docs/decisiones_alcance.md):

1. ``device`` se fuerza a CPU: el lab original elige CUDA si existe, pero el
   baseline debe ser la ejecución en CPU para poder calcular el speedup.
2. Los cachés se construyen en una carpeta nueva en cada corrida; si se
   reutilizaran, la fase de preparación duraría 0 s y no se podría medir.
3. Early stopping desactivado (patience > epochs) para que ambas versiones
   ejecuten exactamente el mismo número de épocas.
4. No se guarda el checkpoint ``.pt`` (escritura a disco ajena al cómputo medido).

Flujo (todo en CPU, un solo proceso para los datos: ``num_workers=0``):

    FASE 1 (CPU, 1 proceso)  build_image_cache: JPEG → RGB → resize 64×64 → uint8
    FASE 2 (CPU)             por lote: __getitem__ (/255, HWC→CHW, augmentation)
                             forward, backward y AdamW con hilos intra-op de PyTorch
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from proyecto3.common.config import BenchmarkConfig
from proyecto3.common.timing import PhaseTimer
from src.neural import (
    NeuralExperiment,
    build_model,
    evaluate_loader,
    make_loaders,
    seed_everything,
)


def _timed_batches(loader, timer: PhaseTimer):
    """Itera el DataLoader midiendo el tiempo que se ESPERA cada lote.

    Con ``num_workers=0`` ese tiempo es el trabajo de CPU de ``__getitem__``
    (abrir el caché, augmentation, /255, transponer) hecho en el proceso
    principal. Se reporta como la sub-métrica ``carga_lotes_cpu``.
    """
    iterator = iter(loader)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        timer.add("carga_lotes_cpu", time.perf_counter() - started)
        yield batch


def run_sequential(
    experiment: NeuralExperiment,
    manifests: dict[str, pd.DataFrame],
    class_names: list[str],
    config: BenchmarkConfig,
    timer: PhaseTimer,
    cache_dir: Path,
) -> dict:
    """Ejecuta un experimento completo en la versión secuencial y devuelve métricas."""
    started_total = time.perf_counter()

    # ---- Configuración idéntica a train_experiment (src/neural.py) -------------
    seed_everything(config.seed)
    if config.threads is not None:          # solo con --threads (medición extra opcional)
        torch.set_num_threads(config.threads)
    else:                                   # comportamiento original del lab
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cpu")            # baseline: CPU (ver diferencia 1)
    model = build_model(
        experiment.architecture,
        len(class_names),
        experiment.dropout,
        experiment.width,
    ).to(device)

    # ======================= FASE 1: PREPARACIÓN (CPU) ==========================
    # make_loaders → build_image_cache para train/validation/test, con un bucle
    # Python secuencial: una imagen a la vez, en un solo núcleo.
    with timer.fase("preparacion_cpu"):
        loaders = make_loaders(
            manifests,
            class_names,
            config.batch_size,
            experiment.augment,
            config.seed,
            cache_dir=cache_dir,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=1e-4,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    # =================== FASE 2: ENTRENAMIENTO (CPU en baseline) ===============
    for epoch in range(1, config.epochs + 1):
        with timer.fase("entrenamiento"):
            model.train()
            loss_sum = 0.0
            observation_count = 0
            correct = 0
            for images, labels in _timed_batches(loaders["train"], timer):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_size_actual = labels.size(0)
                loss_sum += float(loss.item()) * batch_size_actual
                observation_count += batch_size_actual
                correct += int((logits.argmax(dim=1) == labels).sum().item())

        with timer.fase("evaluacion"):
            validation = evaluate_loader(model, loaders["validation"], device)
            epoch_row = {
                "epoch": epoch,
                "train_loss": loss_sum / observation_count,
                "train_accuracy": correct / observation_count,
                "val_accuracy": validation["accuracy"],
                "val_macro_f1": validation["macro_f1"],
            }
            history.append(epoch_row)
            if validation["macro_f1"] > best_f1 + 1e-4:
                best_f1 = validation["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        print(
            f"  [secuencial] {experiment.name} epoch={epoch}/{config.epochs} "
            f"loss={epoch_row['train_loss']:.4f} val_f1={epoch_row['val_macro_f1']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    with timer.fase("evaluacion"):
        model.load_state_dict(best_state)
        test_metrics = evaluate_loader(model, loaders["test"], device)

    timer.add("total", time.perf_counter() - started_total)
    return {
        "device": str(device),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "val_macro_f1": best_f1,
        "epochs_ran": len(history),
        "final_train_loss": history[-1]["train_loss"],
    }
