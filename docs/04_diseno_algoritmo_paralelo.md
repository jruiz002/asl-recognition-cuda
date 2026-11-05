# Paso 4 – Diseño del algoritmo paralelo

## 4.1 Visión general: separación estricta del pipeline

```
 [Disco: 17 400 JPEG 200×200]
          │
 ═════════╪══════════ FASE 1 · PREPARACIÓN DE DATOS · CPU MULTINÚCLEO ═══════════════════
          │  proyecto3/parallel/cpu_prep.py
          │  ProcessPoolExecutor(P = núcleos lógicos), 136 tareas de 128 imágenes
          │  cada proceso: open → decode JPEG → RGB → resize 64×64 → escribe en memmap uint8
          │  (+ experimento con augmentation: DataLoader con W workers que aumentan cada época)
          ▼
 [uint8 N×64×64×3 en RAM]  ── pin_memory() ──►  copia H2D por DMA (non_blocking), 1 B/valor
 ═════════╪══════════ FASE 2 · RED NEURONAL · GPU (CUDA) ════════════════════════════════
          │  proyecto3/parallel/gpu_train.py, gpu_kernels.py
          │  (a) kernel Numba: uint8 HWC → float32 CHW / 255   ← capa de entrada
          │  (b) conv/BN/ReLU/pool/FC con cuDNN y cuBLAS (PyTorch)
          │  (c) CrossEntropy, backward, AdamW
          │  (d) evaluación: forward sin gradientes + argmax en la GPU
          ▼
 [predicciones (2 610 enteros) → CPU → accuracy y macro F1 (sklearn)]
```

**Regla de diseño:** todo lo que toca **archivos** (disco, formatos, decodificación, Pillow) vive en la
FASE 1 en el CPU. Todo lo que es **matemática de la red** vive en la FASE 2 en la GPU. Entre ambas
solo cruza un tensor `uint8` de píxeles crudos.

## 4.2 FASE 1 en CPU: distribución del trabajo entre núcleos

| Elemento | Diseño | Razón |
|---|---|---|
| Unidad de trabajo | Bloque de `CHUNK_SIZE = 128` imágenes | 17 400 / 128 ≈ 136 tareas: suficientes para balancear con 8–32 procesos, sin sobrecosto de una tarea por imagen. |
| Trabajadores | `os.cpu_count()` procesos (`ProcessPoolExecutor`) | Procesos y no hilos: el GIL de Python serializaría a los hilos. |
| Asignación | Dinámica (`pool.map` sobre la cola de tareas) | Un JPEG puede tardar más que otro; el proceso libre toma el siguiente bloque. |
| Salida | Cada proceso escribe en su porción de un `.npy` memmap compartido | Evita serializar 150 MB de píxeles por pickle hacia el proceso principal. |
| Augmentation | `DataLoader(num_workers=min(8, núcleos−1), persistent_workers, prefetch_factor=4, pin_memory)` | Productor-consumidor: hasta 32 lotes listos mientras la GPU entrena. Los workers persisten entre épocas (en Windows cada uno importa torch). |
| Semillas | `worker_init_fn` asigna a cada worker `SafeAugmentation(seed + 1000·(id+1))` | Sin esto, todos los workers copiarían el mismo generador y repetirían transformaciones. |
| Verificación | `tests/test_proyecto3.py` | El caché paralelo es **byte a byte idéntico** al de `build_image_cache` del lab. |

## 4.3 FASE 2 en GPU: kernel propio – hilos, bloques y grid

**Kernel** `normalize_hwc_to_chw_kernel` (`proyecto3/parallel/gpu_kernels.py`, Numba `@cuda.jit`):

```python
x, y, n = cuda.grid(3)                       # x: columna, y: fila, n: imagen
if n < N and y < H and x < W:
    for c in range(C):                       # 3 canales por hilo
        dst[n, c, y, x] = float32(src[n, y, x, c]) / float32(255.0)
```

| Nivel | Configuración | Justificación |
|---|---|---|
| **Hilo** | 1 hilo = 1 píxel (x, y) de 1 imagen; procesa sus 3 canales | Las operaciones son independientes, así que no hace falta sincronizar ni usar memoria compartida. |
| **Bloque** | `THREADS_PER_BLOCK = (16, 16, 1)` = **256 hilos** = 8 warps | Múltiplo de 32 (tamaño de warp). 256 hilos permiten varios bloques activos por SM (alta ocupación). 16 encaja exacto en 64 (sin hilos ociosos). |
| **Grid** | `(⌈W/16⌉, ⌈H/16⌉, N)` = **(4, 4, N)** | Los ejes x/y recorren teselas de 16×16 de la imagen y el eje z recorre las imágenes del lote. |
| Lote de entrenamiento | N = 64 → 4·4·64 = **1 024 bloques × 256 = 262 144 hilos** | Coincide exactamente con 64·64·64 píxeles: un hilo por píxel, sin hilos sobrantes. |
| Validación/test | N = 64 (último lote: 50 o 20) | El eje z se ajusta al tamaño real del lote, sin relleno. |
| Bordes | `if n < N and y < H and x < W` | Protege si H o W no son múltiplos de 16 (probado con 21×19 en los tests). |

**Acceso a memoria.** Dentro de un warp, los hilos consecutivos en `x` **escriben** `float32` contiguos
del plano CHW (escritura coalescente: 16 × 4 B = 64 B por canal y fila) y **leen** bytes con paso 3 (HWC),
que caen en los mismos segmentos de 128 B. La operación está limitada por el ancho de banda de memoria
(1 B leído y 4 B escritos por valor), no por el cómputo. Por eso se mide aparte con
`scripts/bench_kernel.py`.

**Integración con PyTorch sin copias.** `cuda.as_cuda_array(tensor)` usa `__cuda_array_interface__`
para ver la memoria del tensor de PyTorch como un arreglo de Numba. El kernel se lanza en el **mismo
stream** de PyTorch (`cuda.external_stream(torch.cuda.current_stream().cuda_stream)`), así que se ejecuta
en orden con las convoluciones sin sincronizaciones explícitas.

### ¿Por qué la normalización es parte de la red y no "preparación de archivos"?

1. **No toca archivos.** Para cuando el kernel corre, el JPEG ya fue leído, decodificado y redimensionado
   por el CPU. El kernel recibe un tensor numérico en memoria de la GPU.
2. **Es una capa lineal fija del modelo.** `x ↦ x/255` con reordenamiento de ejes equivale a una capa de
   escalado sin parámetros, como `transforms.Normalize` o una `BatchNorm` congelada. Un modelo desplegado
   (por ejemplo, la app de SignBridge) recibiría píxeles `uint8` de la cámara y esta sería su primera
   operación.
3. **Es aritmética elemento a elemento**, el patrón ideal para SIMT, igual que ReLU o BatchNorm, que ya
   corren en la GPU.
4. **Ventaja de transferencia.** Moverla del CPU a la GPU reduce los datos que cruzan PCIe de
   4 B a 1 B por valor. Si se hiciera en el CPU, el CPU haría el trabajo **y además** enviaría 4× más bytes.

Por eso, la frontera CPU/GPU queda exactamente en el **tensor de píxeles crudos**.

## 4.4 FASE 2 en GPU: kernels de PyTorch (cuDNN/cuBLAS)

PyTorch no expone la configuración de bloques de cuDNN, pero el reparto sigue el mismo principio:

| Operación | Cómo se paraleliza en la GPU |
|---|---|
| Conv 3×3 (forward/backward) | cuDNN la transforma en GEMM implícito o Winograd. Cada bloque calcula una tesela de la matriz de salida (canales × píxeles·lote), con memoria compartida y Tensor Cores (TF32) cuando hay. `cudnn.benchmark=True` elige el algoritmo más rápido para 64×64. |
| BatchNorm, ReLU | Kernels elemento a elemento y reducciones por canal (un bloque por canal o tesela). |
| MaxPool | Un hilo por elemento de salida. |
| Lineal | cuBLAS GEMM `[64 × 768] · [768 × 128]`. |
| CrossEntropy | Reducción por fila (log-softmax) y luego por lote. |
| AdamW | `foreach`/multi-tensor: pocos kernels que actualizan todos los parámetros elemento a elemento. |

## 4.5 Transferencias CPU ↔ GPU

| Experimento | Estrategia | Transferencias |
|---|---|---|
| `cnn_reg_drop035` (sin aug) | El conjunto train + validation + test (`uint8`, ≈ 214 MB) se copia **una vez** con `pin_memory()` + `non_blocking`. Los lotes se arman indexando dentro de la GPU con la misma permutación que el `DataLoader` del lab (`epoch_permutation`). | 1 copia por corrida. Por época solo se copian los índices y, al final, 1 escalar de loss y accuracy. |
| `cnn_base_aug` (con aug) | Validation y test residentes como arriba. Train **en streaming**: cada lote `uint8` aumentado sale de un worker, pasa a memoria fijada (`pin_memory=True`) y se copia con `non_blocking=True`. | 191 copias de 786 KB por época, solapadas con el cómputo. Se miden con CUDA events (`h2d_en_bucle`). |

## 4.6 Medición

| Fase (columna en `results/tiempos.csv`) | Qué incluye | Cómo se mide |
|---|---|---|
| `preparacion_cpu` | Decodificación y resize de las 17 400 imágenes (+ carga a RAM) | `perf_counter` |
| `transferencia_h2d` | Copia inicial CPU→GPU (solo en la versión paralela) | `perf_counter` + `cuda.synchronize` |
| `entrenamiento` | Todas las épocas de forward, backward y optimizador (incluye `carga_lotes_cpu` en la secuencial y `kernel_normalizacion` / `h2d_en_bucle` en la paralela) | `perf_counter` + `cuda.synchronize` al abrir y cerrar |
| `evaluacion` | Validación por época + prueba final | igual |
| `kernel_normalizacion` | Suma de las ejecuciones del kernel Numba | `torch.cuda.Event` (sin sincronizar por lote) |
| `total` | Toda la corrida, desde la semilla hasta las métricas de test | `perf_counter` |

Más detalles del procedimiento en `02_medicion_secuencial.md` y `05_implementacion_cuda.md`.
