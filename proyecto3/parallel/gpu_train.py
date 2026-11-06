"""FASE 2 – CONFIGURACIÓN DE CUDA Y ENTRENAMIENTO EN LA GPU.

Aquí solo vive la matemática de la red neuronal: kernel de entrada (Numba),
convoluciones, BatchNorm, ReLU, pooling y capas densas (cuDNN/cuBLAS a través de
PyTorch), la pérdida, el backward y el optimizador AdamW. Nada de este módulo lee
archivos ni decodifica imágenes.

Librerías/funciones usadas para trabajar con la GPU:
- ``torch.cuda.is_available`` / ``torch.device("cuda")``: selección del dispositivo.
- ``torch.backends.cudnn.benchmark = True``: cuDNN prueba los algoritmos de
  convolución disponibles y elige el más rápido para el tamaño fijo 64×64.
- ``Tensor.pin_memory()`` + ``.to(device, non_blocking=True)``: copias H2D por DMA
  desde memoria no paginable, asíncronas respecto al CPU.
- ``numba.cuda`` (``@cuda.jit``, ``as_cuda_array``, ``external_stream``): kernel
  propio de normalización compartiendo memoria y stream con PyTorch.
- ``torch.cuda.Event`` / ``torch.cuda.synchronize``: medición correcta de tiempos.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from proyecto3.common.timing import CudaEventAccumulator
from proyecto3.parallel.gpu_kernels import normalize_torch_batch


def configure_cuda() -> torch.device:
    """Elige la GPU si existe. Si no existe (Mac), avisa y sigue en CPU sin fallar."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        print(f"  [paralelo] GPU detectada: {name} (CUDA {torch.version.cuda})", flush=True)
        return torch.device("cuda")
    print(
        "  [paralelo] AVISO: no hay GPU CUDA. La FASE 2 corre en CPU "
        "(solo válido para pruebas, NO para mediciones).",
        flush=True,
    )
    return torch.device("cpu")


def to_device_uint8(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copia H2D de un arreglo uint8 completo usando memoria fijada (pinned)."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor


def epoch_permutation(generator: torch.Generator, size: int) -> torch.Tensor:
    """Mismo orden de lotes que DataLoader(shuffle=True, generator=g) del lab.

    Por cada época completa, el DataLoader consume del generador: (1) un número para
    la semilla base de los workers, (2) ``torch.randperm`` con el orden de la época y
    (3) al agotarse, ``RandomSampler`` llama OTRA vez a ``randperm`` y descarta el
    resultado (``[: num_samples % n]`` = vacío). Se replican las tres llamadas para que
    ambas versiones vean los lotes en el mismo orden en TODAS las épocas
    (verificado en tests/test_proyecto3.py).
    """
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(size, generator=generator)
    torch.randperm(size, generator=generator)   # llamada descartada al final de la época
    return order


def train_step(model, optimizer, criterion, images, labels, stats) -> None:
    """Un paso de entrenamiento idéntico al del lab (forward, pérdida, backward, AdamW)."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    # Se acumula en la GPU; solo se sincroniza una vez por época (no por lote).
    stats["loss_sum"] += loss.detach() * labels.size(0)
    stats["correct"] += (logits.argmax(dim=1) == labels).sum()
    stats["count"] += labels.size(0)


def new_stats(device: torch.device) -> dict:
    zero = torch.zeros((), device=device)
    return {"loss_sum": zero.clone(), "correct": zero.clone(), "count": 0}


def train_epoch_resident(
    model: nn.Module,
    optimizer,
    criterion,
    images_u8: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    kernel_timer: CudaEventAccumulator,
) -> dict:
    """Época sin augmentation: TODO el conjunto de entrenamiento ya vive en la GPU.

    El CPU solo calcula la permutación (unos pocos KB); los lotes se arman por
    indexación dentro de la GPU, de modo que no hay transferencias CPU→GPU en el bucle.
    """
    device = images_u8.device
    model.train()
    stats = new_stats(device)
    order = epoch_permutation(generator, images_u8.shape[0]).to(device)
    for start in range(0, order.numel(), batch_size):
        index = order[start : start + batch_size]
        kernel_timer.start()
        batch = normalize_torch_batch(images_u8[index])   # kernel Numba en la GPU
        kernel_timer.stop()
        train_step(model, optimizer, criterion, batch, labels[index], stats)
    return stats


def train_epoch_streaming(
    model: nn.Module,
    optimizer,
    criterion,
    loader,
    device: torch.device,
    kernel_timer: CudaEventAccumulator,
    h2d_timer: CudaEventAccumulator,
) -> dict:
    """Época con augmentation: los workers del CPU generan lotes uint8 aumentados y
    la GPU los recibe por PCIe (pinned + non_blocking) mientras sigue calculando."""
    model.train()
    stats = new_stats(device)
    for images_u8, labels in loader:
        h2d_timer.start()
        images_u8 = images_u8.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        h2d_timer.stop()
        kernel_timer.start()
        batch = normalize_torch_batch(images_u8)
        kernel_timer.stop()
        train_step(model, optimizer, criterion, batch, labels, stats)
    return stats


@torch.inference_mode()
def evaluate_resident(
    model: nn.Module,
    images_u8: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    kernel_timer: CudaEventAccumulator,
) -> dict[str, float]:
    """Misma métrica que ``evaluate_loader`` del lab (accuracy y macro F1 con sklearn).

    Las predicciones se calculan en la GPU y se copian al CPU una sola vez al final.
    """
    model.eval()
    predictions = []
    for start in range(0, images_u8.shape[0], batch_size):
        kernel_timer.start()
        batch = normalize_torch_batch(images_u8[start : start + batch_size])
        kernel_timer.stop()
        predictions.append(model(batch).argmax(dim=1))
    predicted = torch.cat(predictions).cpu().numpy()
    true = labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
    }
