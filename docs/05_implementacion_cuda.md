# Paso 5 – Implementación con CUDA Python

## 5.1 Mapa del código

| Archivo | Fase | Contenido |
|---|---|---|
| `proyecto3/sequential/pipeline.py` | Baseline | Bucle original de `train_experiment` con timers. Solo usa funciones de `src/` (lab). |
| `proyecto3/parallel/cpu_prep.py` | **FASE 1 · CPU** | `prepare_splits_parallel`: `ProcessPoolExecutor` + memmap compartido. Solo importa NumPy y Pillow. |
| `proyecto3/parallel/pipeline.py` | FASE 1 → FASE 2 | `run_parallel`: orquesta preparación → H2D → entrenamiento. `AugmentedUint8Dataset` (workers de CPU). |
| `proyecto3/parallel/gpu_kernels.py` | **FASE 2 · GPU** | Kernel Numba `normalize_hwc_to_chw_kernel` + integración zero-copy con PyTorch. |
| `proyecto3/parallel/gpu_train.py` | **FASE 2 · GPU** | `configure_cuda`, `to_device_uint8`, épocas residentes y en streaming, evaluación en la GPU. |
| `proyecto3/common/*` | Ambas | Configuración, rutas portables, `PhaseTimer`, CUDA events, info de hardware y monitores. |
| `benchmark.py` | – | Corre ambas versiones N veces con calentamiento y genera CSV, logs, evidencia y gráficas. |
| `scripts/bench_kernel.py` | – | Speedup propio del kernel frente a NumPy. |
| `scripts/evidencia_fases.py` | – | Mantiene cada fase activa para las capturas de pantalla. |
| `scripts/check_env.py` | – | Verifica la GPU, CUDA, Numba y el dataset; deja `evidence/check_env.txt`. |
| `tests/test_proyecto3.py` | – | Equivalencia: caché, orden de lotes y kernel (simulador y GPU real). |

## 5.2 Funciones y librerías usadas para trabajar con la GPU

| Librería / función | Para qué se usó |
|---|---|
| `torch.cuda.is_available()`, `torch.device("cuda")` | Detectar la GPU y elegir el dispositivo (si no hay GPU, se avisa y se sigue en CPU). |
| `Module.to(device)` | Copiar los pesos del modelo a la memoria de la GPU. |
| `Tensor.pin_memory()` | Memoria del host no paginable: permite DMA directo y copias asíncronas. |
| `Tensor.to(device, non_blocking=True)` | Copias H2D asíncronas, solapables con el cómputo. |
| `DataLoader(pin_memory=True, num_workers=N, persistent_workers=True, prefetch_factor=4)` | Workers de CPU que preparan lotes ya fijados en memoria para la copia rápida. |
| `torch.backends.cudnn.benchmark = True` | cuDNN prueba algoritmos de convolución y elige el más rápido para 64×64. |
| cuDNN (vía `nn.Conv2d`, `nn.BatchNorm2d`, `nn.MaxPool2d`) | Convoluciones, normalización y pooling en la GPU. |
| cuBLAS (vía `nn.Linear`) | Productos de matrices de las capas densas. |
| `torch.optim.AdamW` (implementación `foreach`) | Actualización de parámetros en la GPU con pocos kernels. |
| `numba.cuda.jit` | Compilar el kernel propio de Python a PTX/SASS. |
| `numba.cuda.grid(3)` | Índice global (x, y, z) del hilo dentro del grid. |
| `kernel[blocks_per_grid, threads_per_block, stream]` | Lanzar el kernel con la configuración del paso 4. |
| `numba.cuda.as_cuda_array` / `__cuda_array_interface__` | Compartir memoria de la GPU entre PyTorch y Numba sin copias. |
| `numba.cuda.external_stream` | Ejecutar el kernel en el mismo stream CUDA de PyTorch. |
| `numba.cuda.to_device`, `device_array` | Pruebas del kernel (también en el simulador `NUMBA_ENABLE_CUDASIM=1`). |
| `torch.cuda.synchronize()` | Esperar a la GPU antes de leer el reloj (las operaciones CUDA son asíncronas). |
| `torch.cuda.Event(enable_timing=True)` | Medir el kernel y las copias en el bucle sin sincronizar en cada lote. |
| `torch.cuda.get_device_properties` | Registrar el modelo de GPU, la memoria y los SMs (`results/hw_info.json`). |
| `nvidia-smi --query-gpu=… -l 1` | Registro continuo del uso de núcleos y memoria de la GPU (`evidence/nvidia_smi_*.csv`). |
| `concurrent.futures.ProcessPoolExecutor` | FASE 1: un proceso por núcleo del CPU. |
| `psutil.cpu_percent(percpu=True)` | Evidencia del uso de cada núcleo, etiquetado por fase. |

## 5.3 Optimizaciones respecto al lab, además de mover el cómputo a la GPU

1. **Sin `loss.item()` por lote.** El lab sincroniza CPU↔GPU 191 veces por época al leer la pérdida. La
   versión paralela acumula pérdida y aciertos en la GPU y sincroniza una sola vez por época.
2. **Dataset residente en la GPU** (experimento sin augmentation): 0 transferencias en el bucle.
3. **`uint8` por PCIe** y normalización en la GPU (kernel Numba): 4× menos bytes.
4. **Predicciones en la GPU**: en la evaluación, solo se copia al CPU el vector final de etiquetas.
5. **FASE 1 sin pickle de píxeles**: los procesos escriben directo en un memmap compartido.

## 5.4 Procedimiento de ejecución y medición

1. `python scripts/check_env.py` → evidencia de que se reconoce el hardware (`evidence/check_env.txt`).
2. `python benchmark.py --fresh` → para cada experimento: secuencial (calentamiento + 5 corridas) y luego
   paralela (calentamiento + 5 corridas). En cada corrida se reconstruye la caché y se miden
   `preparacion_cpu`, `transferencia_h2d`, `entrenamiento`, `evaluacion` y `total`, además de las sub-métricas
   `carga_lotes_cpu`, `kernel_normalizacion` y `h2d_en_bucle`.
3. `python scripts/bench_kernel.py` → speedup propio del kernel (20 repeticiones).
4. `python scripts/evidencia_fases.py` → capturas de pantalla de cada fase.

Todos los tiempos de GPU se toman después de `torch.cuda.synchronize()`, así que incluyen el cómputo real
**y las transferencias CPU↔GPU**, como pide el enunciado.

## 5.5 Resultados de la versión paralela

> Se completan con `results/tabla_speedup.md` después de correr en la laptop.

| Experimento | Preparación (s) | H2D (s) | Entrenamiento (s) | Evaluación (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| `cnn_reg_drop035` | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| `cnn_base_aug` | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |

Kernel (`results/kernel_benchmark.md`): CPU NumPy TODO_RESULTADO ms frente a kernel Numba
TODO_RESULTADO ms (TODO_RESULTADO×).
