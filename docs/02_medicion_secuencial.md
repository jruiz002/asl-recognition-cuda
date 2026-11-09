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
3. Se ejecutó `python benchmark.py --fresh` (y, tras dos interrupciones por memoria en la versión paralela,
   se completó sin `--fresh` con `--only`/`--experiments`; ver D12–D14). Para cada experimento:
   - **1 corrida de calentamiento** de 1 época, que no se cuenta (lleva los JPEG a la caché de archivos del
     SO y elimina el costo de la primera importación; ver D7),
   - **N = 5 corridas medidas** completas, cada una reconstruyendo la caché desde los JPEG.
4. En cada corrida se midió el tiempo de pared (`time.perf_counter`) de cada fase:

| Fase | Operaciones (tabla 1.3) |
|---|---|
| `preparacion_cpu` | 2–6: lectura, decodificación, RGB, resize y caché de las 17 400 imágenes |
| `entrenamiento` | 7–16 durante las 6 épocas (incluye `carga_lotes_cpu`: tiempo de `__getitem__`, augmentation, `/255`, CHW) |
| `evaluacion` | 17–19: validación por época + prueba final |
| `total` | Toda la corrida |

5. Mientras tanto, `evidence/cpu_por_nucleo_*.csv` registró el uso de cada núcleo cada 0,25 s.

## 2.3 Resultados

Hardware: **Intel Core i9-13980HX** (24 núcleos físicos: 8 P-cores + 16 E-cores; 32 procesadores lógicos),
**15,6 GB de RAM**, Windows 11, Python 3.12.10, PyTorch 2.14.1 (ver `results/hw_info.json`).

| Experimento | Fase | Media (s) | Desv. estándar (s) | % del total |
|---|---|---:|---:|---:|
| `cnn_reg_drop035` | Preparación (CPU) | 8,14 | 0,44 | 6,9 % |
| | Entrenamiento | 100,76 | 7,69 | 85,7 % |
| | ↳ carga de lotes (CPU) | 5,72 | 0,33 | 4,9 % |
| | Evaluación | 8,61 | 0,54 | 7,3 % |
| | **Total** | **117,53** | 8,49 | 100 % |
| `cnn_base_aug` | Preparación (CPU) | 13,53 | 6,58 | 6,4 % |
| | Entrenamiento | 183,47 | 26,30 | 86,8 % |
| | ↳ carga de lotes (CPU) | 73,88 | 6,70 | 35,0 % |
| | Evaluación | 14,24 | 1,65 | 6,7 % |
| | **Total** | **211,28** | 29,13 | 100 % |

(La "carga de lotes" está **incluida** en el entrenamiento; por eso los porcentajes no suman 100 %.)

Gráfica: `results/figuras/01_tiempos_por_fase.png` (barras azules).
Uso de núcleos: `results/figuras/03_cpu_nucleos_secuencial.png`.

Calidad del modelo (prueba, promedio de las 5 corridas; las 5 dan exactamente el mismo resultado porque el
pipeline en CPU es determinista con semilla 42):

| Experimento | Accuracy | Macro F1 |
|---|---:|---:|
| `cnn_reg_drop035` | 0,9130 | 0,9118 |
| `cnn_base_aug` | 0,7559 | 0,7530 |

## 2.4 Análisis: factores que influyen en el rendimiento

1. **El entrenamiento domina.** Con ~22,7 M MAC por imagen (`cnn_reg`), las 6 épocas suman ~5 × 10¹² MAC.
   En el CPU, con a lo sumo 8 hilos y AVX2, esto representa el **85,7 %** (`cnn_reg_drop035`) y el **86,8 %**
   (`cnn_base_aug`) del tiempo total. Es la fase donde más se puede ganar.
2. **La preparación es secuencial.** Las 17 400 decodificaciones JPEG las hace **un solo proceso**:
   el uso total del CPU durante la preparación es ≈ 1/N: **~8 %** con N = **32** procesadores lógicos, con
   **1,4 núcleos activos** en promedio (> 50 % de uso), según `evidence/EVIDENCIA.md` §2 y
   `results/figuras/03_cpu_preparacion_por_nucleo.png`. El sistema operativo *migra* ese proceso entre núcleos,
   por eso la carga aparece repartida en varias barras bajas y no como un solo núcleo al 100 %. La suma equivale
   a un núcleo y el resto de la capacidad queda ociosa. (Referencia en la Mac de desarrollo, 10 núcleos:
   22 % secuencial frente a 67 % paralela.)
3. **Carga de lotes en el bucle.** Con `num_workers=0`, el cálculo de la red se detiene mientras se arma
   cada lote. En `cnn_base_aug` la augmentation Pillow (73 080 llamadas) agrega **73,9 s** por corrida
   (35 % del total), frente a 5,7 s de carga de lotes en `cnn_reg_drop035` (sin augmentation).
4. **Caché de archivos del SO.** La primera lectura de los JPEG desde disco es más lenta. Por eso se hace el
   calentamiento; sin él, la primera corrida tendría una preparación mayor.
5. **Frecuencia del CPU y temperatura.** En una laptop, el *turbo boost* y el *thermal throttling* hacen
   variar los tiempos. Se refleja en la desviación estándar: **7,2 %** del total en `cnn_reg_drop035` y
   **13,8 %** en `cnn_base_aug` (corridas entre 165,7 s y 236,9 s). Se observó directamente: con el mismo código,
   un calentamiento de validación tardó 16 s en entrenar y la corrida siguiente 35 s, porque el i9 de laptop
   solo sostiene la potencia turbo (PL2) durante ~30–60 s y luego baja al límite sostenido (PL1). Por eso se
   promedian 5 corridas en estado sostenido y se descarta el calentamiento.
   En la preparación de `cnn_base_aug` una corrida tardó 24,9 s frente a ~10–13 s de las demás (desviación
   6,58 s). Como esta fase lee 17 400 archivos, es sensible a factores externos: la caché de archivos de Windows y
   el antivirus instalado (Kaspersky), que revisa cada archivo abierto, son causas probables (no se aislaron).
6. **Hilos intra-op de PyTorch.** El lab limita a 8 hilos. En un i9 con 32 procesadores lógicos, el resto no se
   usa para la matemática de la red: durante el entrenamiento secuencial el CPU total queda en ~32 % con ~9 núcleos
   activos (`evidence/EVIDENCIA.md` §2).
7. **CPU híbrida (P-cores y E-cores).** Los 16 E-cores del i9-13980HX son más lentos; Windows reparte los hilos
   entre ambos tipos, lo que agrega variabilidad.
8. **Tamaño del lote y del modelo.** Lotes de 64 imágenes de 64×64 son pequeños: el CPU pasa una parte del
   tiempo en sobrecosto de Python y de lanzar operaciones, no solo en aritmética.
