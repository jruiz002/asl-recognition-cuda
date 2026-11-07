# Paso 6 – Análisis y conclusiones

> Borrador: los valores `TODO_RESULTADO` salen de `results/tabla_speedup.md`, `results/resumen.csv` y
> `results/kernel_benchmark.md`, generados en la laptop.

## 6.1 Tabla comparativa de tiempos (media ± desviación, N = TODO_RESULTADO corridas)

Hardware: CPU TODO_RESULTADO · GPU TODO_RESULTADO · driver TODO_RESULTADO · CUDA TODO_RESULTADO.

### `cnn_reg_drop035`

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| Transferencia H2D | – | TODO_RESULTADO | – |
| Entrenamiento | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| Evaluación | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| **Total** | TODO_RESULTADO | TODO_RESULTADO | **TODO_RESULTADO×** |

### `cnn_base_aug`

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| Transferencia H2D | – | TODO_RESULTADO | – |
| Entrenamiento | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| Evaluación | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO× |
| **Total** | TODO_RESULTADO | TODO_RESULTADO | **TODO_RESULTADO×** |

Calidad: accuracy/macro F1 de prueba, secuencial TODO_RESULTADO frente a paralela TODO_RESULTADO
(diferencia < TODO_RESULTADO). La aceleración **no** cambió lo que aprende el modelo.

Gráficas: `results/figuras/01_tiempos_por_fase.png`, `02_speedup_por_fase.png`, `05_kernel.png`.

**Speedup** = t_secuencial / t_paralela (por fase, con las medias). La transferencia H2D no existe en la
versión secuencial, así que no tiene speedup: es un **costo agregado** que se incluye en el total paralelo.

## 6.2 ¿La paralelización redujo los tiempos? Causas de las diferencias

**Preparación (CPU multinúcleo).** Se pasó de 1 a TODO_RESULTADO procesos y el speedup fue
TODO_RESULTADO×. No llega al número de núcleos lógicos porque: (a) los núcleos lógicos del
*hyper-threading* comparten unidades de ejecución (el límite real se acerca a los núcleos físicos),
(b) en Windows, crear los procesos (*spawn*) tiene un costo fijo, (c) en las CPU híbridas de Intel los
núcleos de eficiencia (E-cores) son más lentos y (d) la lectura de disco y el acceso a memoria se
comparten (ley de Amdahl: la parte no paralela limita el speedup).

**Transferencia H2D.** TODO_RESULTADO s para ≈ 214 MB en `uint8` (TODO_RESULTADO GB/s). Es pequeña frente
al entrenamiento gracias a la memoria fijada y a transferir 1 byte por valor. En `float32` serían 856 MB.

**Entrenamiento (GPU).** Es la fase con mayor speedup (TODO_RESULTADO×). Las convoluciones 3×3 se
convierten en productos de matrices que cuDNN reparte entre TODO_RESULTADO SMs (miles de núcleos CUDA),
frente a ≤ 8 hilos del CPU. Además: el dataset residente elimina las copias del bucle, la pérdida se
acumula en la GPU (sin 191 sincronizaciones por época) y `cudnn.benchmark` elige el mejor algoritmo.

**Por qué el speedup del entrenamiento no es aún mayor.** El modelo es pequeño (143 K parámetros) y el
lote también (64 × 3 × 64 × 64). Cada kernel tiene poco trabajo, así que la latencia de lanzamiento
(~5–10 µs por kernel, cientos de kernels por paso) y el sobrecosto de Python pesan. El registro de
`nvidia-smi` muestra una utilización de TODO_RESULTADO %, es decir, la GPU no está saturada. Lotes más
grandes o imágenes de mayor resolución aumentarían el speedup.

**`cnn_base_aug`.** La augmentation se repite cada época en el CPU. Con TODO_RESULTADO workers, los lotes
se preparan **en paralelo y solapados** con la GPU. El tiempo de entrenamiento queda limitado por
TODO_RESULTADO (CPU o GPU, según quién sea más lento; se ve en `h2d_en_bucle` y en el uso de la GPU).

**Kernel Numba.** Normalizar 12 180 imágenes: NumPy TODO_RESULTADO ms frente a kernel TODO_RESULTADO ms
(TODO_RESULTADO×). Incluyendo la copia, "H2D uint8 + kernel" (TODO_RESULTADO ms) le gana a
"normalizar en el CPU + H2D float32" (TODO_RESULTADO ms) porque mueve 4× menos bytes por PCIe.

## 6.3 Discusión técnica

### ¿Por qué es más eficiente procesar las imágenes con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red a la GPU?

Porque son dos cargas de trabajo con perfiles opuestos, y cada procesador está diseñado para una de ellas:

| | Manejo de archivos e imágenes | Matemática de la red |
|---|---|---|
| Naturaleza | E/S de disco, llamadas al SO, decodificación JPEG (Huffman de longitud variable, ramificaciones) | Productos de matrices y operaciones elemento a elemento, regulares y sin ramificación |
| Paralelismo | Pocas tareas grandes e independientes (1 por archivo) | Millones de operaciones idénticas (1 por elemento de salida) |
| Procesador ideal | **CPU**: núcleos complejos con predicción de saltos, cachés grandes y acceso al SO y al disco | **GPU**: miles de núcleos simples SIMT, gran ancho de banda de memoria, Tensor Cores |
| En la GPU sería... | Imposible (la GPU no accede al disco) o ineficiente (divergencia de warps en la decodificación) | – |
| En un solo núcleo del CPU sería... | Lento pero correcto: lo que hacía el lab | Muy lento: ≤ 8 hilos para ~10 TFLOP |

Además, la separación crea un **pipeline**: mientras la GPU entrena con el lote *k*, los workers del CPU
preparan el lote *k+1*, así que ningún recurso queda ocioso. Ver `results/figuras/03_cpu_nucleos_*.png`:
en la versión paralela todos los núcleos se ocupan durante la preparación y, durante el entrenamiento,
el CPU queda casi libre mientras la GPU trabaja (`04_gpu_uso.png`).

### ¿Cómo permitió CUDA reducir o cambiar el tiempo de procesamiento?

- **Cambió la distribución del tiempo:** en la versión secuencial, el entrenamiento era el TODO_RESULTADO %
  del total. En la paralela, el entrenamiento bajó a TODO_RESULTADO s y ahora la preparación y la
  transferencia pesan proporcionalmente más (TODO_RESULTADO %). Es la ley de Amdahl en acción.
- **Redujo el tiempo total** de TODO_RESULTADO s a TODO_RESULTADO s (TODO_RESULTADO×).
- Mecanismos: miles de hilos sobre convoluciones (cuDNN), GEMM (cuBLAS), kernel propio de normalización
  con un hilo por píxel, datos residentes en la memoria de la GPU y copias asíncronas desde memoria fijada.

### ¿Qué factores justifican o limitan el uso de CUDA?

**Lo justifican:**
- Alta intensidad aritmética: ~10 TFLOP por experimento en operaciones regulares.
- El entrenamiento se repite (épocas, experimentos, búsqueda de hiperparámetros): el costo fijo de CUDA
  (contexto, copia inicial) se amortiza.
- Los datos caben completos en la memoria de la GPU (≈ 214 MB en `uint8`).
- Ecosistema maduro: PyTorch, cuDNN y cuBLAS dan la aceleración casi gratis; Numba permite kernels propios
  en Python.

**Lo limitan:**
- **Costos fijos:** inicializar el contexto CUDA, compilar con JIT (Numba) y la transferencia H2D. Con
  pocos datos (`--sample`) la versión paralela puede ser **más lenta**.
- **Problema pequeño:** imágenes de 64×64 y batch 64 no saturan la GPU (utilización TODO_RESULTADO %).
- **Ancho de banda PCIe:** cada byte que cruza tiene costo; por eso se transfiere `uint8` y una sola vez.
- **Partes no paralelizables en la GPU:** la E/S y la decodificación siguen en el CPU (Amdahl).
- **Hardware y portabilidad:** requiere una GPU NVIDIA, drivers y versiones compatibles (por ejemplo,
  TensorFlow ya no soporta CUDA en Windows nativo). El SVM-HOG (liblinear) no tiene versión CUDA.
- **Determinismo:** los kernels de la GPU suman en distinto orden, así que hay diferencias numéricas
  mínimas frente al CPU (sin impacto en la accuracy).

## 6.4 Conclusiones

1. TODO_RESULTADO (speedup total y por fase).
2. La separación CPU (archivos) / GPU (matemática) aprovecha cada procesador en lo que hace mejor y permite
   solapar ambos trabajos.
3. El mayor beneficio está en el entrenamiento; la preparación se acelera con multiprocesamiento,
   pero queda limitada por la E/S y el costo de crear procesos.
4. CUDA conviene cuando el cómputo es grande y repetido. Para cargas pequeñas, los costos fijos dominan.
