# Proyecto 3 – Mejorando algoritmos con CUDA: reconocimiento ASL

**CC3069 Computación Paralela y Distribuida · UVG · Semestre II 2026**

Aceleración del *Laboratorio 3 de Data Science* (clasificador de letras ASL con CNN en PyTorch):
la **preparación de imágenes** pasa a usar **todos los núcleos del CPU** y la **matemática de la red
neuronal** se delega a la **GPU con CUDA** (PyTorch/cuDNN + un kernel propio en Numba).

```
[JPEG en disco] ─► FASE 1 · CPU multinúcleo ─► uint8 ─► H2D ─► FASE 2 · GPU (CUDA) ─► métricas
                  ProcessPoolExecutor              pinned      kernel Numba /255 + CHW
                  decode + resize 64×64                        conv/BN/FC (cuDNN, cuBLAS)
                  DataLoader workers (augment.)                backward + AdamW
```

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Código **original secuencial** del lab (sin modificar) | `src/`, `entrenar_modelos.py`, `analisis_exploratorio.py`, `evaluar_fotos.py` |
| Versión secuencial **instrumentada** (baseline) | `proyecto3/sequential/pipeline.py` |
| **FASE 1 – preparación de datos (CPU multinúcleo)** | `proyecto3/parallel/cpu_prep.py` |
| **FASE 2 – configuración CUDA y entrenamiento (GPU)** | `proyecto3/parallel/gpu_train.py`, `proyecto3/parallel/gpu_kernels.py` |
| Orquestación de la versión paralela | `proyecto3/parallel/pipeline.py` |
| Benchmark (ambas versiones, N corridas) | `benchmark.py` |
| Notebook de entrega | `notebooks/proyecto3_entrega.ipynb` |
| Análisis (pasos 1–6) y decisiones | `docs/` |
| Resultados y gráficas | `results/` |
| Evidencia (logs de nvidia-smi, uso por núcleo, capturas) | `evidence/` |
| Instalación en Windows · día de la laptop | `SETUP_WINDOWS.md` · `CHECKLIST_LAPTOP.md` |

## Uso rápido

```bash
python scripts/check_env.py                              # verifica GPU, CUDA, Numba y dataset
python benchmark.py --fresh                              # benchmark completo (5 corridas por versión)
python benchmark.py --sample 20 --epochs 1 --runs 1      # prueba rápida
python scripts/bench_kernel.py                           # speedup del kernel Numba
python scripts/evidencia_fases.py --seconds 40           # mantiene cada fase activa para las capturas
python scripts/make_plots.py                             # regenera tablas y gráficas
python -m unittest discover -s tests -v                  # pruebas (equivalencia secuencial/paralela)
NUMBA_ENABLE_CUDASIM=1 python -m unittest tests.test_proyecto3   # kernel en el simulador (sin GPU)
```

Flags de `benchmark.py`: `--runs`, `--epochs`, `--sample`, `--only {sequential,parallel,both}`,
`--experiments`, `--threads`, `--workers`, `--no-warmup`, `--data-root`, `--fresh`, `--no-plots`.

## Documentación

1. [Análisis de la solución secuencial](docs/01_analisis_secuencial.md)
2. [Medición del tiempo secuencial](docs/02_medicion_secuencial.md)
3. [Oportunidades de paralelización](docs/03_oportunidades_paralelizacion.md)
4. [Diseño del algoritmo paralelo (hilos, bloques, grid)](docs/04_diseno_algoritmo_paralelo.md)
5. [Implementación con CUDA Python](docs/05_implementacion_cuda.md)
6. [Análisis y conclusiones](docs/06_analisis_conclusiones.md)
- [Decisiones de alcance y de medición](docs/decisiones_alcance.md)

## Integrantes y contribuciones

| Integrante | Usuario GitHub | Contribución principal |
|---|---|---|
| **José Ruiz** | `jruiz002` | Configuración base, arquitectura común y monitoreo de hardware, FASE 1 CPU multinúcleo con ProcessPoolExecutor, harness de benchmark y pruebas de integración. |
| **Dilary Cruz** | `dils1809` | Análisis exploratorio del dataset, recolección de fotos externas de validación, scripts de diagnóstico y evaluación, y FASE 2 de entrenamiento GPU con PyTorch/cuDNN. |
| **Gerardo Fernández** | `GerardoFdez7` | Modelos secuenciales base, análisis de algoritmos secuenciales, diseño de bloques/grids CUDA, kernel CUDA con Numba y herramientas de visualización de métricas. |


---

# Laboratorio 3 original (base del proyecto)

## SignBridge: reconocimiento de alfabeto ASL
Proyecto Python para explorar el conjunto ASL Alphabet, preparar particiones reproducibles y definir los experimentos de clasificación de imágenes de manos.

## Estructura

```text
sign-language-recognition/
├── analisis_exploratorio.py   # Inventario, gráficos y particiones
├── entrenar_modelos.py        # CNN, MLP, SVM y transformaciones
├── evaluar_fotos.py           # Pruebas oficiales y por integrante
├── src/                       # Datos, preprocesamiento y modelos
├── tests/                     # Pruebas automatizadas
├── data/external/             # Fotografías aportadas por el equipo
├── artifacts/                 # Métricas, figuras y pesos generados
└── requirements.txt
```

`data/raw`, `data/processed` y `artifacts` se excluyen de Git por su tamaño.

## Instalación

```powershell
py -m pip install -r requirements.txt
```

## 1. Análisis y particiones

Con los datos en `data/raw`:

```powershell
py analisis_exploratorio.py
```

También puede indicarse `--data-dir "C:\ruta\al\dataset"` o solicitar la descarga con `--download`.

## 2. Entrenamiento y comparación

```powershell
py entrenar_modelos.py
```

Se entrenan dos CNN con variaciones de hiperparámetros, dos MLP, SVM-HOG con dos valores de C y versiones con transformaciones. Los resultados existentes se reutilizan; `--force` obliga a repetirlos.

Resultados generados:

- `data/manifests/`: entrenamiento, validación y prueba.
- `artifacts/models/`: estados de PyTorch y SVM.
- `artifacts/metrics/`: historiales, predicciones y métricas.
- `artifacts/figures/`: exploración, comparación y matrices de confusión.

## 3. Fotografías externas

Coloque al menos cinco letras distintas por persona en `data/external/<integrante>/`, con nombres como `A_01.jpg`, y ejecute:

```powershell
py evaluar_fotos.py
```

La etiqueta debe aparecer al inicio del nombre. Se aceptan A–Z, `del`, `nothing` y `space`.

## Pruebas

```powershell
py -m unittest discover -s tests -v
```

Las pruebas crean imágenes sintéticas, comprueban cabeceras y particiones, validan HOG y verifican las dimensiones de las tres arquitecturas neuronales.

## Configuración reproducible

- Semilla global: `42`.
- Submuestra: `600` imágenes por clase, 17 400 en total.
- Particiones: 70 % entrenamiento, 15 % validación y 15 % prueba.
- Entrada: RGB de `64 × 64`, tipo `float32`, valores en `[0, 1]`.
- Selección: macro F1 de validación; la prueba se consulta después.
- Mejor resultado actual: SVM-HOG con C=1, macro F1 de prueba `0.9443`.

Fuente: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
