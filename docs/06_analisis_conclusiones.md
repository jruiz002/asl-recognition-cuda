# Paso 6 – Análisis y conclusiones

Fuentes: `results/tabla_speedup.md`, `results/resumen.csv`, `results/kernel_benchmark.md` y
`evidence/EVIDENCIA.md` (uso de hardware por fase), generados en la laptop.

## 6.1 Tabla comparativa de tiempos (media ± desviación, N = 5 corridas)

Hardware: CPU **Intel Core i9-13980HX** (24 núcleos / 32 hilos, 15,6 GB de RAM) · GPU **NVIDIA GeForce RTX 4060
Laptop** (8 GB, 24 SMs, compute capability 8.9) · driver **555.97** · CUDA **12.6** (PyTorch 2.14.1, cuDNN 9.10).
Mismos datos (17 400 imágenes), 6 épocas, batch 64, semilla 42, 1 calentamiento no contabilizado.

### `cnn_reg_drop035`

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | 8,14 ± 0,44 | 4,25 ± 0,59 | 1,91× |
| Transferencia H2D | – | 0,048 ± 0,004 | – |
| Entrenamiento | 100,76 ± 7,69 | 6,49 ± 1,19 | **15,52×** |
| Evaluación | 8,61 ± 0,54 | 0,46 ± 0,05 | **18,60×** |
| **Total** | **117,53 ± 8,49** | **11,28 ± 1,80** | **10,42×** |

### `cnn_base_aug`

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | 13,53 ± 6,58 | 4,98 ± 0,33 | 2,72× |
| Transferencia H2D | – | 0,043 ± 0,000 | – |
| Entrenamiento | 183,47 ± 26,30 | 32,06 ± 3,71 | **5,72×** |
| Evaluación | 14,24 ± 1,65 | 0,92 ± 0,35 | **15,50×** |
| **Total** | **211,28 ± 29,13** | **38,04 ± 4,22** | **5,55×** |

Calidad del modelo (conjunto de prueba, promedio de las 5 corridas):

| Experimento | Secuencial: accuracy / macro F1 | Paralela: accuracy / macro F1 |
|---|---:|---:|
| `cnn_reg_drop035` | 0,9130 / 0,9118 | 0,9455 / 0,9451 |
| `cnn_base_aug` | 0,7559 / 0,7530 | 0,7329 / 0,7300 |

Las diferencias (≤ 3,3 puntos, en ambos sentidos) no vienen de la aceleración en sí: la versión paralela forzada a
CPU reproduce **exactamente** la pérdida y el F1 de la secuencial (D16). En la GPU, cuDNN suma en otro orden
(la suma de punto flotante no es asociativa) y en `cnn_base_aug` cada worker usa su propio generador de
augmentation; con solo 6 épocas, esas pequeñas diferencias cambian la trayectoria del entrenamiento. La
aceleración **no** empeoró sistemáticamente lo que aprende el modelo.

Gráficas: `results/figuras/01_tiempos_por_fase.png`, `02_speedup_por_fase.png`, `05_kernel.png`.

**Speedup** = t_secuencial / t_paralela (por fase, con las medias). La transferencia H2D no existe en la
versión secuencial, así que no tiene speedup: es un **costo agregado** que se incluye en el total paralelo.

## 6.2 ¿La paralelización redujo los tiempos? Causas de las diferencias

Sí: el tiempo total bajó de **117,5 s a 11,3 s (10,4×)** en `cnn_reg_drop035` y de **211,3 s a 38,0 s (5,6×)** en
`cnn_base_aug`.

**Preparación (CPU multinúcleo).** Se pasó de 1 a **32 procesos** y el speedup fue **1,9×–2,7×**. El uso del CPU sí
cambió radicalmente: de **~8 % con 1,4 núcleos activos** a **~72 % con ~25 de 32 núcleos activos**
(`evidence/EVIDENCIA.md` §2, capturas 03 y 04). El speedup no llega a 32 porque:
(a) los 32 hilos son 8 P-cores con *hyper-threading* + 16 E-cores más lentos;
(b) en Windows, crear 32 procesos (*spawn*, cada uno reimporta Python, NumPy y Pillow) cuesta ~0,5 s fijos;
(c) las imágenes de 200×200 son pequeñas, así que abrir cada archivo (sistema de archivos, antivirus) pesa tanto
como decodificarlo, y el disco y la memoria se comparten;
(d) la fase incluye partes en serie: reservar el archivo `.npy` y leer los conjuntos que irán a la GPU.
Es la ley de Amdahl: la parte no paralelizable limita el speedup.

**Transferencia H2D.** **0,048 s** para ≈ 214 MB en `uint8` (~4,5 GB/s, con memoria fijada) en `cnn_reg_drop035`.
Es el 0,4 % del tiempo paralelo. En `float32` serían 856 MB: transferir 1 byte por valor y una sola vez la mantiene
despreciable.

**Entrenamiento (GPU).** Es la fase con mayor speedup en `cnn_reg_drop035`: **15,5×** (100,8 s → 6,5 s). Las
convoluciones 3×3 se convierten en productos de matrices que cuDNN reparte entre los **24 SMs** (3 072 núcleos
CUDA), frente a ≤ 8 hilos del CPU. Además: el dataset residente elimina las copias del bucle (H2D en el bucle:
0 s), la pérdida se acumula en la GPU (una sincronización por época en vez de 191) y `cudnn.benchmark` elige el
mejor algoritmo. nvidia-smi registró un **GPU-Util medio de 77 % (máx. 98 %)** durante esta fase, con el CPU en
~13 % (captura 10).

**Evaluación (GPU).** **15,5×–18,6×**: es inferencia pura (sin backward) sobre datos ya residentes en la GPU, y las
predicciones se copian al CPU una sola vez.

**Por qué el speedup del entrenamiento no es aún mayor.** El modelo es pequeño (143 K parámetros) y el lote
también (64 × 3 × 64 × 64). Cada kernel tiene poco trabajo, así que la latencia de lanzamiento (~5–10 µs por kernel,
cientos de kernels por paso) y el sobrecosto de Python pesan. Lotes más grandes o imágenes de mayor resolución
aumentarían el speedup.

**`cnn_base_aug`: el cuello de botella pasa al CPU.** La augmentation (rotaciones, recortes, cambios de color con
Pillow) se repite **en cada época** y solo puede hacerse en el CPU. En la versión secuencial cuesta 73,9 s por
corrida (35 % del total). En la paralela la hacen **4 workers** solapados con la GPU, pero la GPU los procesa más
rápido de lo que el CPU los produce: nvidia-smi registró solo **~14 % de GPU-Util medio** en esta fase (captura 12),
frente a 77 % en `cnn_reg_drop035`. Por eso el entrenamiento acelera 5,7× y no 15×. Se usaron 4 workers y no 8
porque, con 16 GB de RAM, 8 procesos con torch agotaron la memoria (D13): **la RAM limitó el paralelismo del CPU,
y el CPU limitó a la GPU.**

**Kernel Numba.** Normalizar 12 180 imágenes: NumPy **503,9 ms** frente a kernel **3,2 ms (157×)**. Incluyendo la
copia, "H2D `uint8` + kernel" (**16,1 ms**) le gana a "normalizar en el CPU + H2D `float32`" (**809,2 ms**) por
**50×**, porque mueve 4× menos bytes por PCIe. Además, el kernel propio (3,2 ms) es más rápido que la misma operación
con PyTorch nativo en la GPU (13,7 ms), porque hace la conversión, la división y la transposición en una sola pasada.

## 6.3 Discusión técnica

### ¿Por qué es más eficiente procesar las imágenes con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red a la GPU?

Porque son dos cargas de trabajo con perfiles opuestos, y cada procesador está diseñado para una de ellas:

| | Manejo de archivos e imágenes | Matemática de la red |
|---|---|---|
| Naturaleza | E/S de disco, llamadas al SO, decodificación JPEG (Huffman de longitud variable, ramificaciones) | Productos de matrices y operaciones elemento a elemento, regulares y sin ramificación |
| Paralelismo | Pocas tareas grandes e independientes (1 por archivo) | Millones de operaciones idénticas (1 por elemento de salida) |
| Procesador ideal | **CPU**: núcleos complejos con predicción de saltos, cachés grandes y acceso al SO y al disco | **GPU**: miles de núcleos simples SIMT, gran ancho de banda de memoria |
| En la GPU sería... | Imposible (la GPU no accede al disco) o ineficiente (divergencia de warps en la decodificación) | – |
| En un solo núcleo del CPU sería... | Lento pero correcto: lo que hacía el lab | Muy lento: ≤ 8 hilos para ~10 TFLOP |

Los datos lo confirman. Repartir los archivos entre los 32 hilos del CPU subió el uso del CPU de ~8 % a ~72 % y
aceleró la preparación ~2×; esa fase **no** se habría podido mandar a la GPU. Mandar la matemática a la GPU aceleró
el entrenamiento 15,5× y dejó al CPU casi libre (~13 %) mientras la GPU trabajaba al 77 %.

Además, la separación crea un **pipeline**: mientras la GPU entrena con el lote *k*, los workers del CPU preparan el
lote *k+1*, así que ningún recurso queda ocioso sin necesidad. En la captura 10 se ve: ráfagas en los 32 núcleos
del CPU al empezar cada corrida (FASE 1) y la GPU al 98 % entre ráfagas (FASE 2).

### ¿Cómo permitió CUDA reducir o cambiar el tiempo de procesamiento?

- **Redujo el tiempo total** de 117,5 s a 11,3 s (**10,4×**) en `cnn_reg_drop035` y de 211,3 s a 38,0 s (**5,6×**) en
  `cnn_base_aug`.
- **Cambió la distribución del tiempo:** en la versión secuencial, el entrenamiento era el **85,7 %** del total
  (`cnn_reg_drop035`). En la paralela, el entrenamiento bajó a 6,5 s y ahora la preparación y la transferencia
  pesan proporcionalmente mucho más: **38 %** del total (4,3 s de 11,3 s), frente a 6,9 % antes. Es la ley de Amdahl
  en acción: al acelerar la parte dominante, la siguiente fase pasa a ser el límite.
- **Mecanismos:** miles de hilos sobre las convoluciones (cuDNN) y las capas densas (cuBLAS), el kernel propio de
  normalización con un hilo por píxel (bloques de 16×16, grid de 4×4×N), los datos residentes en la memoria de la
  GPU y las copias asíncronas desde memoria fijada.

**Funciones y librerías adicionales para trabajar con la GPU:**

| Librería / función | Uso en el proyecto |
|---|---|
| `torch.cuda.is_available`, `torch.device("cuda")`, `.to(device)` | Detección y selección de la GPU |
| cuDNN (`torch.backends.cudnn.benchmark = True`) | Convoluciones, BatchNorm y pooling; elige el algoritmo más rápido |
| cuBLAS (vía PyTorch) | Capas lineales (GEMM) |
| `Tensor.pin_memory()` + `.to(device, non_blocking=True)` | Copias H2D por DMA, asíncronas |
| `torch.cuda.Event`, `torch.cuda.synchronize` | Medición correcta de tiempos en la GPU |
| `numba.cuda` (`@cuda.jit`, `cuda.grid`, `as_cuda_array`, `external_stream`) | Kernel propio de normalización, compartiendo memoria (zero-copy) y stream con PyTorch |
| `numba-cuda` + `cuda-toolkit` 12.6 (NVVM, nvJitLink por pip) | Compilar el kernel sin instalar el CUDA Toolkit |
| `nvidia-smi` | Monitoreo y registro del uso de la GPU (evidencia) |
| En el CPU: `concurrent.futures.ProcessPoolExecutor`, `numpy.memmap`, `DataLoader(num_workers, persistent_workers, pin_memory)`, `psutil` | Preparación multinúcleo, memoria compartida entre procesos y monitoreo por núcleo |

### ¿Qué factores justifican o limitan el uso de CUDA?

**Lo justifican:**
- Alta intensidad aritmética: ~10 TFLOP por experimento en operaciones regulares (convoluciones y GEMM). Es justo
  donde se obtuvo 15,5× (entrenamiento) y 18,6× (evaluación).
- El entrenamiento se repite (épocas, experimentos, búsqueda de hiperparámetros): el costo fijo de CUDA (contexto,
  copia inicial de 0,05 s) se amortiza.
- Los datos caben completos en la memoria de la GPU (≈ 214 MB en `uint8` de 8 GB).
- Ecosistema maduro: PyTorch, cuDNN y cuBLAS dan la aceleración casi gratis; Numba permite kernels propios en
  Python.

**Lo limitan (todos observados en este proyecto):**
- **Costos fijos:** inicializar el contexto CUDA, compilar con JIT (Numba) y la transferencia H2D. Con pocos datos
  (`--sample 20`, 1 época) la versión paralela fue **más lenta** en `cnn_base_aug` (9,4 s frente a 2,3 s), porque
  arrancar los workers y la GPU no se amortiza.
- **Problema pequeño:** imágenes de 64×64 y batch 64 no saturan la GPU de forma continua.
- **Etapas que siguen en el CPU:** la E/S, la decodificación y la augmentation no se pueden llevar a la GPU (sin
  reescribirlas). En `cnn_base_aug` el CPU es el cuello de botella: la GPU queda al ~14 % de uso (Amdahl).
- **Memoria del sistema:** con 16 GB de RAM se pasó de 8 a 4 workers para no agotar la memoria (D13).
- **Ancho de banda PCIe:** cada byte que cruza tiene costo; por eso se transfiere `uint8` y una sola vez.
- **Compatibilidad de versiones:** driver, toolkit, PyTorch y Numba deben coincidir. En la laptop hubo que usar
  PyTorch `cu126` por el driver 555.97 y alinear Numba con el nvJitLink 12.6 de torch (D12). TensorFlow ya no
  soporta CUDA en Windows nativo, y el SVM-HOG (liblinear) no tiene versión CUDA.
- **Determinismo:** la GPU suma en distinto orden, así que hay diferencias numéricas frente al CPU (≤ 3,3 puntos de
  F1 con 6 épocas, en ambos sentidos).

## 6.4 Conclusiones

1. **CUDA redujo el tiempo total 10,4×** en `cnn_reg_drop035` (117,5 s → 11,3 s) y **5,6×** en `cnn_base_aug`
   (211,3 s → 38,0 s). Por fase: preparación 1,9×–2,7× (CPU multinúcleo), entrenamiento 15,5× y 5,7×, evaluación
   18,6× y 15,5× (GPU).
2. La separación CPU (archivos) / GPU (matemática) aprovecha cada procesador en lo que hace mejor y permite
   solapar ambos trabajos. Los monitores lo confirman fase por fase: 32 hilos del CPU en la preparación, GPU al
   77–98 % y CPU casi libre en el entrenamiento.
3. El mayor beneficio está en el entrenamiento y la evaluación. La preparación se acelera con multiprocesamiento,
   pero queda limitada por la E/S, el costo de crear procesos en Windows y los núcleos híbridos.
4. Al acelerar la parte dominante, el cuello de botella se desplaza (Amdahl): en `cnn_reg_drop035` la preparación
   pasó de 7 % a 38 % del tiempo, y en `cnn_base_aug` la augmentation en el CPU limita a la GPU.
5. CUDA conviene cuando el cómputo es grande y repetido. Para cargas pequeñas, los costos fijos dominan, y la
   memoria del sistema y la compatibilidad de versiones son restricciones prácticas reales.
