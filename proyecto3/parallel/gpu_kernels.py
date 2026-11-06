"""Kernel CUDA propio (Numba ``@cuda.jit``): CAPA DE ENTRADA DE LA RED EN LA GPU.

Qué hace: convierte un lote ``uint8 [N, H, W, C]`` (tal como sale del CPU) en el
tensor ``float32 [N, C, H, W]`` normalizado a [0, 1] que espera la primera capa
convolucional de PyTorch. Es la misma operación que el lab hacía en el CPU dentro de
``ASLImageDataset.__getitem__`` (``/255`` + ``transpose(2, 0, 1)``).

Por qué pertenece a la FASE 2 (GPU) y no a la preparación de archivos
(docs/04_diseno_algoritmo_paralelo.md): no toca archivos ni decodifica nada. Es
una transformación aritmética elemento a elemento sobre un tensor que ya está en
memoria, equivalente a una capa de normalización/escalado al inicio del modelo.
Además, hacerla en la GPU permite transferir ``uint8`` (1 byte por valor) en vez de
``float32`` (4 bytes), lo que reduce a la cuarta parte el tráfico por PCIe.

Organización de hilos, bloques y grid (cuda.grid(3)):

    bloque = (16, 16, 1)  → 256 hilos; cada hilo es UN píxel (x, y) de UNA imagen
                             y procesa sus 3 canales.
    grid   = (ceil(W/16), ceil(H/16), N) → para 64×64: (4, 4, N)
             eje x/y del grid: teselas de 16×16 de la imagen
             eje z del grid:   una imagen del lote por cada capa z
    Lote de 64 imágenes → 4·4·64 = 1 024 bloques · 256 hilos = 262 144 hilos
                         = 64·64·64 píxeles (exactamente un hilo por píxel).

Acceso a memoria: hilos consecutivos en x leen píxeles consecutivos (lectura
casi coalescente, con paso de 3 bytes) y ESCRIBEN floats consecutivos del plano
CHW (escritura 100 % coalescente: 16 hilos × 4 B = 64 B contiguos por canal).

Prueba sin GPU (Mac): ``NUMBA_ENABLE_CUDASIM=1`` ejecuta el kernel en el simulador.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32

THREADS_PER_BLOCK = (16, 16, 1)   # 256 hilos: múltiplo de 32 (warp), buena ocupación
DIVISOR_255 = float32(255.0)


@cuda.jit
def normalize_hwc_to_chw_kernel(src, dst):
    """src: uint8[N, H, W, C]  →  dst: float32[N, C, H, W] = src / 255."""
    x, y, n = cuda.grid(3)          # x: columna, y: fila, n: índice de imagen
    if n < src.shape[0] and y < src.shape[1] and x < src.shape[2]:
        for c in range(src.shape[3]):
            dst[n, c, y, x] = float32(src[n, y, x, c]) / DIVISOR_255


def blocks_per_grid(n_images: int, height: int, width: int) -> tuple[int, int, int]:
    """Número de bloques en cada eje para cubrir todos los píxeles de todas las imágenes."""
    return (
        math.ceil(width / THREADS_PER_BLOCK[0]),
        math.ceil(height / THREADS_PER_BLOCK[1]),
        n_images,
    )


def launch_normalize(src_device, dst_device, stream=0) -> None:
    """Lanza el kernel sobre arreglos que YA están en la GPU (o en el simulador)."""
    n_images, height, width, _ = src_device.shape
    grid = blocks_per_grid(n_images, height, width)
    normalize_hwc_to_chw_kernel[grid, THREADS_PER_BLOCK, stream](src_device, dst_device)


def normalize_numpy(batch_uint8: np.ndarray) -> np.ndarray:
    """Referencia en CPU (NumPy): la misma operación que hacía el lab en __getitem__."""
    return (batch_uint8.astype(np.float32) / np.float32(255.0)).transpose(0, 3, 1, 2).copy()


def numba_cuda_available() -> bool:
    """True si hay GPU NVIDIA real (o simulador activo) para Numba."""
    try:
        return bool(cuda.is_available())
    except Exception:   # noqa: BLE001 - drivers ausentes lanzan errores variados
        return False


# ------------------------- Integración con PyTorch ---------------------------
def normalize_torch_batch(batch_uint8):
    """Recibe un ``torch.Tensor`` uint8 [N,H,W,C] y devuelve float32 [N,C,H,W].

    - En CUDA: lanza el kernel Numba SIN copias, compartiendo memoria con PyTorch
      a través de ``__cuda_array_interface__`` (``cuda.as_cuda_array``), y en el
      MISMO stream que usa PyTorch (``cuda.external_stream``), así el orden queda
      garantizado sin sincronizar.
    - Sin CUDA (Mac): misma operación con PyTorch en CPU (solo para pruebas).
    """
    import torch

    n_images, height, width, channels = batch_uint8.shape
    if batch_uint8.is_cuda:
        batch_uint8 = batch_uint8.contiguous()
        output = torch.empty(
            (n_images, channels, height, width),
            dtype=torch.float32,
            device=batch_uint8.device,
        )
        stream = cuda.external_stream(torch.cuda.current_stream().cuda_stream)
        launch_normalize(cuda.as_cuda_array(batch_uint8), cuda.as_cuda_array(output), stream)
        return output
    return batch_uint8.to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
