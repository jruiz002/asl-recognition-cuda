# Decisiones de alcance y de medición

El enunciado pide documentar *"los datos utilizados, los procedimientos y las decisiones tomadas"*.
Esta es la bitácora de decisiones, con su justificación.

## D1. Framework: PyTorch, sin migración

El lab ya usa **PyTorch** (`src/neural.py`). PyTorch trae CUDA nativo en Windows, a diferencia de
TensorFlow ≥ 2.11, que en Windows nativo no usa la GPU. Por eso **no fue necesario WSL2 ni cambiar el modelo**.

## D2. Experimentos medidos: 2 de los 9 neuronales

| Experimento | Por qué se eligió |
|---|---|
| `cnn_reg_drop035` | Es la CNN con **más cómputo convolucional** (6 capas conv, 22,7 M MAC por imagen; ver `01`). Es el mejor caso para la GPU y representa la "matemática de la red". Sin augmentation: su preparación es solo decodificación y resize. |
| `cnn_base_aug` | Tiene **augmentation por imagen en cada época** (73 080 transformaciones Pillow). Es la carga de CPU que se repite durante el entrenamiento y permite mostrar el CPU multinúcleo (workers del DataLoader) trabajando **en paralelo** con la GPU. |

Entre los dos se cubren **las dos fases** del pipeline (preparación en CPU y entrenamiento en GPU) y los
dos patrones de transferencia (dataset residente en la GPU y streaming).

**Por qué no los 9:** el tiempo en la laptop con GPU es muy limitado. La versión secuencial en CPU de
cada experimento tarda varios minutos (medido: ~2,0 min por corrida en `cnn_reg_drop035` y ~3,5 min en
`cnn_base_aug`, es decir ~10 y ~18 min por experimento solo en la versión secuencial) y el protocolo exige
5 repeticiones + calentamiento para ambas versiones. Los 9 experimentos multiplicarían ese tiempo
por ~4,5. Los otros 7 usan las mismas operaciones (convoluciones, capas lineales, la misma augmentation)
y no agregan información nueva sobre la paralelización. `--runs` es configurable (5 por defecto, se puede
bajar a 3) y `--experiments` permite agregar cualquier otro.

## D3. SVM-HOG fuera del alcance de CUDA

`LinearSVC` (liblinear) es un solver de **CPU sin implementación CUDA**, y el SVM no es una red
neuronal. La rúbrica pide delegar "la matemática de la red neuronal" a la GPU. Su extracción de HOG
(`src/classical.py`) es paralelizable por imagen en el CPU con el mismo patrón de `cpu_prep.py`, pero se
deja en el CPU y se menciona en el análisis como fuera de alcance.

## D4. Baseline = código original del lab, forzado a CPU

- Se reutilizan las funciones originales sin modificarlas (`build_image_cache`, `ASLImageDataset`,
  `make_loaders`, `build_model`, `evaluate_loader`). El bucle de `train_experiment` se reproduce
  línea por línea en `proyecto3/sequential/pipeline.py`, solo con timers.
- `device = cpu` forzado. El lab elige `cuda` si existe; el baseline tiene que ser la ejecución en CPU.
- Se conserva `num_workers=0` (un solo proceso para los datos) y `torch.set_num_threads(min(8, …))`,
  igual que el lab. **Nota honesta:** la matemática de PyTorch en CPU ya usa hasta 8 hilos internos. El
  baseline es "secuencial" en el manejo de datos y "la ejecución original del lab" en el cómputo.
  `--threads 1` permite medir además una versión estrictamente de 1 hilo, si hay tiempo.

## D5. Épocas fijas (early stopping desactivado durante el benchmark)

Con early stopping (`patience=2`), cada corrida podría tener un número de épocas distinto según
pequeñas diferencias numéricas (CPU frente a GPU), y los tiempos dejarían de ser comparables. En el
benchmark se usa `patience = epochs + 1`: ambas versiones hacen **exactamente 6 épocas**. La lógica
de selección del mejor estado por macro F1 de validación se conserva.

## D6. Caché reconstruida en cada corrida

El lab reutiliza `data/processed/*.npy` si existe. En el benchmark cada corrida usa una carpeta nueva
(`data/bench_cache/…`, que se borra al terminar), así **la fase de preparación se mide siempre**.

## D7. Calentamiento (warm-up) de 1 época

Antes de las corridas medidas de cada versión/experimento se hace una corrida de **1 época** que no se
cuenta. Sirve para: (1) crear el contexto CUDA (~1–3 s), (2) compilar el kernel Numba (JIT),
(3) que cuDNN elija sus algoritmos y (4) dejar los JPEG en la **caché de archivos del sistema operativo**.
El punto (4) también aplica al secuencial, así ambas versiones leen del disco en las mismas condiciones.

## D8. Mismos datos y mismas condiciones

| Condición | Cómo se garantiza |
|---|---|
| Mismas imágenes | El mismo `data/manifests/all.csv` del lab. Las rutas se remapean a la máquina local (`proyecto3/common/paths.py`). |
| Mismos píxeles | El caché paralelo es byte a byte igual al original (test automático). |
| Misma normalización | El kernel produce bit a bit lo mismo que `float32/255` de NumPy (test en simulador y en GPU). |
| Mismo orden de lotes | `epoch_permutation` reproduce la permutación del `DataLoader` del lab (test automático). |
| Mismo modelo e inicialización | `seed_everything(42)` + `build_model` del lab. |
| Mismos hiperparámetros | lr, dropout, weight decay, batch 64, 6 épocas. |
| Mismo hardware | Ambas versiones corren en la **misma laptop**, en la misma ejecución de `benchmark.py`. |

**Verificación empírica:** si se fuerza la versión paralela a correr la FASE 2 en el CPU (Mac, sin GPU),
`cnn_reg_drop035` produce **exactamente el mismo macro F1 de prueba** que la versión secuencial después de
3 épocas. La pérdida reportada solo difiere en el 7.º decimal, porque se acumula en float32 en vez de
float64. Esto demuestra que las dos versiones hacen el mismo cálculo. (Un primer intento difería desde
la época 2: `RandomSampler` hace una llamada extra a `randperm` al agotarse cada época. Se detectó con un
trazado paso a paso y se corrigió en `epoch_permutation`.)

Diferencias inevitables, que no afectan la validez de la comparación de tiempos: (a) la GPU usa kernels
de punto flotante con distinto orden de suma, así que la pérdida puede variar en el 3.er–4.º decimal;
(b) en `cnn_base_aug`, cada worker usa su propio generador de augmentation, por lo que las transformaciones
siguen la misma distribución pero no son idénticas imagen por imagen. Por eso se reporta la accuracy y el
macro F1 de ambas versiones, para mostrar que la calidad del modelo es equivalente.

## D9. Lo que no se mide

- `torch.save` del checkpoint, los CSV de predicciones y las gráficas del lab (E/S de reportes, no cómputo).
- El cierre de los workers del DataLoader al final de la corrida (es limpieza de procesos del SO; en
  macOS tarda ~5 s por worker). Se imprime en el log como "no contabilizado".

## D10. Kernel Numba como capa de entrada (ver `04`, sección 4.3)

Se eligió normalizar en la GPU porque es la primera operación de la red, no manejo de archivos. Además,
reduce 4× los bytes por PCIe. Su speedup propio frente a NumPy se mide aparte con `scripts/bench_kernel.py`.

## D11. Entorno

- Windows: Python 3.12, PyTorch con CUDA (wheel oficial de pytorch.org) y `numba-cuda[cu12]`
  (el target CUDA mantenido por NVIDIA trae NVVM por pip y no necesita instalar el CUDA Toolkit).
- Mac (desarrollo): Python 3.13 en `.venv`. El kernel se valida con `NUMBA_ENABLE_CUDASIM=1`. No se usa MPS.

## D12. Ajustes de entorno en la laptop real (Windows 11, RTX 4060 Laptop, driver 555.97)

Ninguno cambia la lógica; están en `requirements-windows.txt`:

- **PyTorch `cu126` en vez de `cu128`:** el driver 555.97 soporta hasta CUDA 12.5 y los wheels `cu128` piden un
  driver ≥ 570. No se actualizó el driver porque la laptop era prestada.
- **NumPy < 2.5:** NumPy 2.5 eliminó `np.row_stack`, que numba-cuda 0.30.4 usa al compilar cualquier kernel.
- **Piezas CUDA de Numba en la serie 12.6:** torch carga su propio nvJitLink 12.6 y el NVVM 12.9 de Numba
  producía `nvJitLinkError ... may need newer version of nvJitLink`. Con todo en 12.6, torch y Numba conviven
  en el mismo proceso (zero-copy verificado por `check_env.py` y los tests).
- El repositorio se clonó **fuera de OneDrive** (`C:\p3`) para que la sincronización no interfiriera con la
  caché del benchmark ni con los tiempos.

## D13. Memoria: OpenBLAS en los procesos hijos y `--loader-workers 4`

La laptop tiene **15,6 GB de RAM** y, con los programas abiertos, ~17 GB de memoria ya comprometida antes de
empezar. Se midió la memoria comprometida del sistema durante el benchmark:

| Situación | Memoria comprometida | Efecto |
|---|---:|---|
| 32 procesos de preparación, NumPy por defecto | **+23,2 GB** | Windows aborta: "El archivo de paginación es demasiado pequeño" |
| 32 procesos con `OPENBLAS_NUM_THREADS=1` | **+0,8 GB** | Funciona, y la preparación pasa de 3,85 s a 2,36 s |
| 8 workers del DataLoader (cada uno importa torch, ~0,9 GB) | 20,5 → 27,7 GB (RAM libre: 0,75 GB) | Un worker muere: `WinError 6` / `DataLoader worker exited unexpectedly` |
| 4 workers (`--loader-workers 4`) | ~3,6 GB menos | Termina sin errores |

- Cada proceso hijo reservaba búferes de OpenBLAS para 32 hilos aunque decodificar JPEG no usa álgebra lineal.
  `proyecto3/parallel/cpu_prep.py` fija `OPENBLAS_NUM_THREADS=1` antes de crear los procesos (no afecta al
  proceso principal ni a los hilos de PyTorch).
- La versión paralela de `cnn_base_aug` se midió con **4 workers** de augmentation en lugar de 8. Es un ejemplo
  concreto de un factor que limita el paralelismo: la RAM disponible.

## D14. Benchmark completado en partes

Las dos interrupciones de D13 ocurrieron en la versión paralela. Como las corridas completas se guardan una por
una en `results/tiempos.csv`, no se repitieron: se completó con `--only parallel --experiments …` sin `--fresh`,
con el mismo hardware, datos e hiperparámetros. Las 4 ejecuciones (`results/logs/`) suman exactamente 5 corridas
medidas por versión y experimento. `scripts/armar_evidencia.py` junta todas las ejecuciones e ignora las
muestras de las corridas interrumpidas.

## D15. Captura 12 en una corrida extra

Para tener tiempo de capturar el entrenamiento paralelo con augmentation, se hizo una corrida extra de
`cnn_base_aug` con 12 épocas. **No forma parte de los promedios:** su fila y sus archivos se eliminaron de
`results/` y `evidence/` después de la captura.

## D16. Validación de corrección en la laptop

- La versión paralela **forzada a CPU** (`CUDA_VISIBLE_DEVICES=-1`) reproduce exactamente la pérdida de cada
  época y el F1 de la secuencial (2,8764 → 2,0477; F1 0,2622): la lógica es idéntica.
- En la GPU las métricas difieren algunas décimas (por ejemplo F1 0,9451 frente a 0,9118 en `cnn_reg_drop035`;
  0,7300 frente a 0,7530 en `cnn_base_aug`): cuDNN suma en otro orden (punto flotante no asociativo) y, en
  `cnn_base_aug`, cada worker usa su propio generador de augmentation. Con solo 6 épocas, esas diferencias
  cambian la trayectoria del entrenamiento; no indican un error.
