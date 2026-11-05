# Paso 2 – Medición del tiempo de ejecución secuencial

## 2.1 Qué se midió

La versión secuencial es el **pipeline original del laboratorio** (`proyecto3/sequential/pipeline.py`,
que llama a las funciones de `src/neural.py`), ejecutado en el **CPU** de la laptop de pruebas, con:

- la partición original de 17 400 imágenes (12 180 / 2 610 / 2 610),
- los experimentos `cnn_reg_drop035` y `cnn_base_aug` (justificación en `decisiones_alcance.md`, D2),
- 6 épocas fijas, batch 64, semilla 42,
- `num_workers=0` (un proceso para los datos) y los hilos intra-op de PyTorch del lab (`min(8, …)`).

## 2.2 Procedimiento

1. Se verificó el entorno con `python scripts/check_env.py` (`evidence/check_env.txt`).
2. Se cerraron las demás aplicaciones y la laptop se conectó al cargador en modo de máximo
   rendimiento (los portátiles bajan la frecuencia con batería).
3. Se ejecutó `python benchmark.py --fresh`. Para cada experimento:
   - **1 corrida de calentamiento** de 1 época, que no se cuenta (lleva los JPEG a la caché de archivos del
     SO y elimina el costo de la primera importación; ver D7),
   - **N = TODO_RESULTADO corridas medidas** completas, cada una reconstruyendo la caché desde los JPEG.
4. En cada corrida se midió el tiempo de pared (`time.perf_counter`) de cada fase:

| Fase | Operaciones (tabla 1.3) |
|---|---|
| `preparacion_cpu` | 2–6: lectura, decodificación, RGB, resize y caché de las 17 400 imágenes |
| `entrenamiento` | 7–16 durante las 6 épocas (incluye `carga_lotes_cpu`: tiempo de `__getitem__`, augmentation, `/255`, CHW) |
| `evaluacion` | 17–19: validación por época + prueba final |
| `total` | Toda la corrida |

5. Mientras tanto, `evidence/cpu_por_nucleo_*.csv` registró el uso de cada núcleo cada 0,25 s.

## 2.3 Resultados

Hardware: TODO_RESULTADO (CPU, núcleos físicos/lógicos, RAM; ver `results/hw_info.json`).

| Experimento | Fase | Media (s) | Desv. estándar (s) | % del total |
|---|---|---:|---:|---:|
| `cnn_reg_drop035` | Preparación (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | Entrenamiento | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | ↳ carga de lotes (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | Evaluación | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | **Total** | TODO_RESULTADO | TODO_RESULTADO | 100 % |
| `cnn_base_aug` | Preparación (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | Entrenamiento | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | ↳ carga de lotes (CPU) | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | Evaluación | TODO_RESULTADO | TODO_RESULTADO | TODO_RESULTADO |
| | **Total** | TODO_RESULTADO | TODO_RESULTADO | 100 % |

Gráfica: `results/figuras/01_tiempos_por_fase.png` (barras azules).
Uso de núcleos: `results/figuras/03_cpu_nucleos_secuencial.png`.

Calidad del modelo (verifica que el baseline reproduce el lab): accuracy de prueba TODO_RESULTADO,
macro F1 TODO_RESULTADO.

## 2.4 Análisis: factores que influyen en el rendimiento

1. **El entrenamiento domina.** Con ~22,7 M MAC por imagen (`cnn_reg`), las 6 épocas suman ~5 × 10¹² MAC.
   En el CPU, con a lo sumo 8 hilos y AVX2/AVX-512, esto representa TODO_RESULTADO % del tiempo total.
2. **La preparación es secuencial.** Las 17 400 decodificaciones JPEG las hace **un solo proceso**:
   el uso total del CPU durante la preparación es ≈ 1/N (TODO_RESULTADO % con N = TODO_RESULTADO núcleos lógicos;
   ver `results/figuras/03_cpu_preparacion_por_nucleo.png`). El sistema operativo *migra* ese proceso entre núcleos,
   por eso la carga aparece repartida en varias barras bajas y no como un solo núcleo al 100 %. La suma equivale
   a un núcleo y el resto de la capacidad queda ociosa. (Referencia en la Mac de desarrollo, 10 núcleos:
   22 % secuencial frente a 67 % paralela.)
3. **Carga de lotes en el bucle.** Con `num_workers=0`, el cálculo de la red se detiene mientras se arma
   cada lote. En `cnn_base_aug` la augmentation Pillow (73 080 llamadas) agrega TODO_RESULTADO s por corrida.
4. **Caché de archivos del SO.** La primera lectura de los JPEG desde disco es más lenta. Por eso se hace el
   calentamiento; sin él, la primera corrida tendría una preparación mayor.
5. **Frecuencia del CPU y temperatura.** En una laptop, el *turbo boost* y el *thermal throttling* hacen
   variar los tiempos. Se refleja en la desviación estándar (TODO_RESULTADO %).
6. **Hilos intra-op de PyTorch.** El lab limita a 8 hilos. En un i9 con más núcleos, el resto no se usa
   para la matemática de la red.
7. **Tamaño del lote y del modelo.** Lotes de 64 imágenes de 64×64 son pequeños: el CPU pasa una parte del
   tiempo en sobrecosto de Python y de lanzar operaciones, no solo en aritmética.
