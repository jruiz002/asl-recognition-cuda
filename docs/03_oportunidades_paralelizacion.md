# Paso 3 – Identificación de oportunidades de paralelización

Se analizó cada operación del paso 1 (`01_analisis_secuencial.md`, tabla 1.3) con tres preguntas:
**¿es independiente entre datos?**, **¿es aritmética regular o tiene E/S y ramificación?** y
**¿cuántos datos mueve en relación con lo que calcula?** (intensidad aritmética).

## 3.1 Clasificación de las operaciones

| Op. | Operación | Paralelismo disponible | ¿Dónde conviene? | Justificación |
|---|---|---|---|---|
| 2–6 | Lectura, decodificación JPEG, RGB y resize | Por **archivo** (17 400 tareas independientes) | **CPU multinúcleo** | Requiere E/S de disco y llamadas al SO, que la GPU no puede hacer. La decodificación Huffman es secuencial dentro de cada archivo y con mucha ramificación (mala para SIMT). Cada archivo es independiente: hay paralelismo perfecto entre núcleos. |
| 9 | Augmentation (Pillow) | Por **imagen** (73 080 tareas) | **CPU multinúcleo** (workers del DataLoader) | Pillow es código C para CPU. Repartir lotes entre procesos permite preparar el siguiente lote mientras la GPU entrena el actual (solapamiento productor-consumidor). |
| 7, 11 | Permutación y ensamblado del lote | Bajo | CPU (o GPU por indexación) | Es poco trabajo. Si el dataset ya está en la GPU, el lote se arma indexando en memoria del dispositivo, sin copias. |
| 10 | Normalización `/255` + HWC→CHW | Por **elemento** (12 288 valores × imagen) | **GPU – kernel propio (Numba)** | Aritmética trivial, totalmente independiente y regular: un hilo por píxel. Hacerla en la GPU permite transferir `uint8` (1 B) en vez de `float32` (4 B): **4× menos tráfico PCIe**. Es la capa de entrada de la red (ver paso 4). |
| 12 | Forward (conv, BN, ReLU, pool, FC) | Por **elemento de salida**, canal e imagen | **GPU (cuDNN / cuBLAS)** | Son ~22,7 M MAC por imagen: un producto de matrices gigantesco (im2col / Winograd). Tiene alta intensidad aritmética y reutiliza datos en memoria compartida; es justo para lo que se diseñaron los miles de núcleos CUDA. |
| 13 | CrossEntropy | Por fila del lote | GPU | Reducciones paralelas sobre `[64, 29]`. Al quedarse en la GPU se evita copiar los logits al CPU. |
| 14 | Backward | Igual que el forward (≈ 2× su costo) | **GPU (cuDNN / cuBLAS)** | Gradientes respecto de activaciones y pesos: convoluciones y GEMM de la misma forma. |
| 15 | AdamW | Por **parámetro** (143 177 a 815 805) | GPU | Actualización elemento a elemento independiente. PyTorch la fusiona en pocos kernels (`foreach`). |
| 16 | Acumular loss y accuracy | Reducción | GPU, con una sola sincronización por época | El lab hace `loss.item()` en cada lote, lo que obliga a sincronizar CPU↔GPU 191 veces por época. Al acumular en la GPU solo se sincroniza una vez. |
| 17, 19 | Validación y test | Por imagen | GPU (forward) + CPU (métricas sklearn) | Forward sin gradientes en la GPU; solo se copia al CPU el vector final de predicciones (2 610 enteros). |
| 18 | `deepcopy(state_dict)` | – | GPU→GPU | Se copia dentro de la memoria de la GPU. |
| 20 | Persistencia, gráficas | – | CPU | Es E/S y no es parte del cómputo medido. |
| HOG + SVM | Descriptor + `LinearSVC` | Por imagen | CPU (fuera de alcance) | liblinear no tiene implementación CUDA y el SVM no es la red neuronal. Ver `decisiones_alcance.md`. |

## 3.2 Operaciones seleccionadas para la GPU

1. **Forward, backward y optimizador** (ops. 12–15) con PyTorch + CUDA (cuDNN para convoluciones y
   BatchNorm, cuBLAS para las capas lineales). Concentran más del 90 % del cómputo aritmético
   (~10 TFLOP por experimento).
2. **Normalización de entrada** (op. 10) con un **kernel CUDA propio escrito en Numba**. Muestra de forma
   explícita la organización de hilos, bloques y grid (paso 4) y reduce 4× los bytes por PCIe.
3. **Residencia de datos en la GPU:** en el experimento sin augmentation, el conjunto completo en `uint8`
   (12 180 × 12 288 B ≈ 150 MB, más 32 MB por cada una de validation y test) se copia **una sola vez**.
   Durante las 6 épocas ya no hay transferencias CPU→GPU en el bucle.

## 3.3 Operaciones que se quedan en el CPU (multinúcleo)

1. **Lectura + decodificación + resize** con `ProcessPoolExecutor`, un proceso por núcleo lógico.
2. **Augmentation por época** con `DataLoader(num_workers=N, persistent_workers=True, pin_memory=True)`.
   Los workers producen lotes `uint8` en paralelo mientras la GPU consume los anteriores.

## 3.4 Beneficio esperado y riesgos

| Fase | Beneficio esperado | Riesgo / límite |
|---|---|---|
| Preparación (CPU) | Speedup cercano al número de núcleos físicos, limitado por el disco y el costo de crear procesos (*spawn* en Windows). | Con pocos datos (`--sample`), crear el pool domina el tiempo (Amdahl). |
| Transferencia H2D | Nueva en la versión paralela (no existe en la secuencial). Con `uint8` + memoria fijada es pequeña (~200 MB a varios GB/s). | Es un costo adicional que el speedup del entrenamiento debe compensar. |
| Entrenamiento | El mayor speedup: miles de núcleos CUDA frente a ≤ 8 hilos del CPU. | Modelo y lote pequeños (64×64, batch 64) → la GPU puede quedar subutilizada y la latencia de lanzar kernels pesa (ver paso 6). |
| Evaluación | Speedup alto por el forward en la GPU. | Las métricas de sklearn siguen en el CPU (costo pequeño). |
