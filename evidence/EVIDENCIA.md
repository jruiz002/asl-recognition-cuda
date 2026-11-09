# Evidencia de ejecución – Proyecto 3 (ASL + CUDA)

Documento generado automáticamente por `scripts/armar_evidencia.py` el 2026-11-08 23:08.
Ejecuciones del benchmark incluidas: `20261107_234430`, `20261108_002048`, `20261108_222830`, `20261108_225815` (cada una con su log, su CSV de CPU por núcleo y su CSV de
nvidia-smi con la misma marca de tiempo). Si hay más de una, es porque el benchmark oficial se completó en partes
sobre el mismo `results/tiempos.csv`, con el mismo hardware, datos e hiperparámetros.

Objetivo (rúbrica, 30 pts): demostrar con **logs** y **capturas de los monitores de recursos** que el entorno
reconoció y usó el hardware asignado **en cada fase**:

| Fase | Hardware asignado | Dónde se demuestra |
|---|---|---|
| FASE 1 · Preparación de imágenes (leer JPEG, decodificar, resize) | **CPU multinúcleo** | §2 (tabla por fase), capturas 03–04, figura de núcleos |
| FASE 2 · Entrenamiento y evaluación de la red | **GPU NVIDIA (CUDA)** | §2, §3 (`GPU detectada`), capturas 05–07 y 10–12, figura de GPU |
| Baseline secuencial | **CPU** (sin GPU) | §2 (GPU-Util ≈ 0 %), captura 09 |

---

## 1. Hardware y software detectados

| Componente | Valor |
|---|---|
| CPU | 13th Gen Intel(R) Core(TM) i9-13980HX |
| Núcleos físicos | 24 |
| Procesadores lógicos | 32 |
| RAM (GB) | 15.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Memoria GPU (GB) | 8.0 |
| Multiprocesadores (SM) | 24 |
| Compute capability | 8.9 |
| Driver NVIDIA | 555.97 |
| CUDA (PyTorch) | 12.6 |
| cuDNN | 91002 |
| PyTorch | 2.14.1+cu126 |
| Numba | 0.68.1 |
| numba-cuda | 0.30.4 |
| NumPy | 2.4.6 |
| Python | 3.12.10 |
| Sistema operativo | Windows 11 (AMD64) |

**Salida de check_env.py** · [`evidence/check_env.txt`](check_env.txt)

```text
{
  "fecha": "2026-11-07T23:11:38",
  "sistema": "Windows 11 (AMD64)",
  "python": "3.12.10",
  "cpu": "13th Gen Intel(R) Core(TM) i9-13980HX",
  "nucleos_fisicos": 24,
  "nucleos_logicos": 32,
  "ram_gb": 15.6,
  "torch": "2.14.1+cu126",
  "torch_hilos_intraop": 24,
  "cuda_disponible": true,
  "cuda_version_torch": "12.6",
  "cudnn": 91002,
  "numba": "0.68.1",
  "numba_cuda": "0.30.4",
  "numpy": "2.4.6",
  "pillow": "11.3.0",
  "pandas": "2.3.3",
  "psutil": "7.2.2",
  "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
  "gpu_memoria_gb": 8.0,
  "gpu_sms": 24,
  "gpu_compute_capability": "8.9",
  "driver_nvidia": "555.97",
  "gpu_smi": "NVIDIA GeForce RTX 4060 Laptop GPU"
}

[OK]   CPU: 13th Gen Intel(R) Core(TM) i9-13980HX · 24 núcleos físicos / 32 lógicos
[OK]   PyTorch ve la GPU: NVIDIA GeForce RTX 4060 Laptop GPU · CUDA 12.6 · cuDNN 91002 · matmul de prueba = 101274.8
[OK]   Kernel Numba compilado y verificado (== NumPy) en NVIDIA GeForce RTX 4060 Laptop GPU
[OK]   Interoperabilidad Numba ↔ PyTorch (zero-copy) verificada
[OK]   Dataset y manifiesto: 17,400 imágenes encontradas ({'test': 2610, 'train': 12180, 'validation': 2610})
[OK]   psutil disponible (monitor de núcleos)
[OK]   nvidia-smi disponible

TODO LISTO para el benchmark.
```

## 2. Uso de hardware por fase (a partir de los logs de monitoreo)

#### Ejecución `20261107_234430`

Fuente: `evidence/cpu_por_nucleo_20261107_234430.csv` (uso de los 32 procesadores lógicos cada 0,25 s, etiquetado con la fase activa) cruzado por hora con `evidence/nvidia_smi_20261107_234430.csv` (nvidia-smi cada 1 s).
Solo corridas medidas y terminadas (sin calentamiento ni corridas interrumpidas). Núcleos activos = procesadores lógicos con uso > 50 %.

| Versión | Modelo | Fase | Hardware esperado | CPU total medio | Núcleos activos (medio / máx) | GPU-Util medio / máx | Memoria GPU máx |
|---|---|---|---|---:|---:|---:|---:|
| secuencial | cnn_reg_drop035 | FASE 1 · Preparación de imágenes | CPU, 1 proceso | 8.1 % | 1.4 / 12 | 0 % / 0 % | 91 MiB |
| secuencial | cnn_reg_drop035 | FASE 2 · Entrenamiento | CPU (hilos de PyTorch), GPU sin uso | 32.4 % | 9.1 / 22 | 0 % / 0 % | 91 MiB |
| secuencial | cnn_reg_drop035 | FASE 2 · Evaluación | CPU, GPU sin uso | 31.3 % | 8.9 / 14 | 0 % / 0 % | 91 MiB |

> Nota: en fases muy cortas (p. ej. la transferencia H2D, < 0,1 s) puede no haber muestras; nvidia-smi muestrea cada 1 s, así que en fases de pocos segundos el GPU-Util es aproximado. La memoria de la GPU es la de TODA la tarjeta (incluye unos cientos de MiB del escritorio de Windows y del contexto CUDA que queda de corridas anteriores); lo que demuestra el uso de CUDA es el GPU-Util.

#### Ejecución `20261108_002048`

Fuente: `evidence/cpu_por_nucleo_20261108_002048.csv` (uso de los 32 procesadores lógicos cada 0,25 s, etiquetado con la fase activa) cruzado por hora con `evidence/nvidia_smi_20261108_002048.csv` (nvidia-smi cada 1 s).
Solo corridas medidas y terminadas (sin calentamiento ni corridas interrumpidas). Núcleos activos = procesadores lógicos con uso > 50 %.

| Versión | Modelo | Fase | Hardware esperado | CPU total medio | Núcleos activos (medio / máx) | GPU-Util medio / máx | Memoria GPU máx |
|---|---|---|---|---:|---:|---:|---:|
| paralela | cnn_reg_drop035 | FASE 1 · Preparación de imágenes | CPU, todos los núcleos | 72.4 % | 24.8 / 32 | 4 % / 88 % | 527 MiB |
| paralela | cnn_reg_drop035 | Transferencia CPU → GPU | Bus PCIe (CPU → GPU) | 35.2 % | 9.0 / 16 | 59 % / 91 % | 527 MiB |
| paralela | cnn_reg_drop035 | FASE 2 · Entrenamiento | GPU (CUDA); CPU libre o con workers de augmentation | 13.3 % | 3.3 / 31 | 77 % / 98 % | 527 MiB |
| paralela | cnn_reg_drop035 | FASE 2 · Evaluación | GPU (CUDA) | 10.6 % | 2.9 / 5 | 68 % / 97 % | 527 MiB |

> Nota: en fases muy cortas (p. ej. la transferencia H2D, < 0,1 s) puede no haber muestras; nvidia-smi muestrea cada 1 s, así que en fases de pocos segundos el GPU-Util es aproximado. La memoria de la GPU es la de TODA la tarjeta (incluye unos cientos de MiB del escritorio de Windows y del contexto CUDA que queda de corridas anteriores); lo que demuestra el uso de CUDA es el GPU-Util.

#### Ejecución `20261108_222830`

Fuente: `evidence/cpu_por_nucleo_20261108_222830.csv` (uso de los 32 procesadores lógicos cada 0,25 s, etiquetado con la fase activa) cruzado por hora con `evidence/nvidia_smi_20261108_222830.csv` (nvidia-smi cada 1 s).
Solo corridas medidas y terminadas (sin calentamiento ni corridas interrumpidas). Núcleos activos = procesadores lógicos con uso > 50 %.

| Versión | Modelo | Fase | Hardware esperado | CPU total medio | Núcleos activos (medio / máx) | GPU-Util medio / máx | Memoria GPU máx |
|---|---|---|---|---:|---:|---:|---:|
| secuencial | cnn_base_aug | FASE 1 · Preparación de imágenes | CPU, 1 proceso | 7.9 % | 2.3 / 10 | 0 % / 0 % | 0 MiB |
| secuencial | cnn_base_aug | FASE 2 · Entrenamiento | CPU (hilos de PyTorch), GPU sin uso | 33.1 % | 11.1 / 24 | 0 % / 0 % | 0 MiB |
| secuencial | cnn_base_aug | FASE 2 · Evaluación | CPU, GPU sin uso | 36.1 % | 11.9 / 27 | 0 % / 0 % | 0 MiB |

> Nota: en fases muy cortas (p. ej. la transferencia H2D, < 0,1 s) puede no haber muestras; nvidia-smi muestrea cada 1 s, así que en fases de pocos segundos el GPU-Util es aproximado. La memoria de la GPU es la de TODA la tarjeta (incluye unos cientos de MiB del escritorio de Windows y del contexto CUDA que queda de corridas anteriores); lo que demuestra el uso de CUDA es el GPU-Util.

#### Ejecución `20261108_225815`

Fuente: `evidence/cpu_por_nucleo_20261108_225815.csv` (uso de los 32 procesadores lógicos cada 0,25 s, etiquetado con la fase activa) cruzado por hora con `evidence/nvidia_smi_20261108_225815.csv` (nvidia-smi cada 1 s).
Solo corridas medidas y terminadas (sin calentamiento ni corridas interrumpidas). Núcleos activos = procesadores lógicos con uso > 50 %.

| Versión | Modelo | Fase | Hardware esperado | CPU total medio | Núcleos activos (medio / máx) | GPU-Util medio / máx | Memoria GPU máx |
|---|---|---|---|---:|---:|---:|---:|
| paralela | cnn_base_aug | FASE 1 · Preparación de imágenes | CPU, todos los núcleos | 65.1 % | 21.9 / 32 | 0 % / 20 % | 333 MiB |
| paralela | cnn_base_aug | Transferencia CPU → GPU | Bus PCIe (CPU → GPU) | 23.0 % | 8.0 / 8 | 14 % / 14 % | 333 MiB |
| paralela | cnn_base_aug | FASE 2 · Entrenamiento | GPU (CUDA); CPU libre o con workers de augmentation | 25.1 % | 9.1 / 25 | 14 % / 99 % | 333 MiB |
| paralela | cnn_base_aug | FASE 2 · Evaluación | GPU (CUDA) | 18.4 % | 5.9 / 10 | 18 % / 66 % | 333 MiB |

> Nota: en fases muy cortas (p. ej. la transferencia H2D, < 0,1 s) puede no haber muestras; nvidia-smi muestrea cada 1 s, así que en fases de pocos segundos el GPU-Util es aproximado. La memoria de la GPU es la de TODA la tarjeta (incluye unos cientos de MiB del escritorio de Windows y del contexto CUDA que queda de corridas anteriores); lo que demuestra el uso de CUDA es el GPU-Util.

## 3. Logs del benchmark

Log completo: [`results/logs/benchmark_20261107_234430.log`](../results/logs/benchmark_20261107_234430.log). Extracto (detección de hardware, procesos usados y tiempos por corrida):

```text
  "torch_hilos_intraop": 24,
SECUENCIAL · cnn_reg_drop035 · calentamiento
  tiempos (s): preparacion_cpu=78.96 · transferencia_h2d=0.00 · entrenamiento=20.81 · evaluacion=3.15 · total=105.13
SECUENCIAL · cnn_reg_drop035 · corrida 1/5
  tiempos (s): preparacion_cpu=8.90 · transferencia_h2d=0.00 · entrenamiento=111.62 · evaluacion=8.83 · total=129.40
SECUENCIAL · cnn_reg_drop035 · corrida 2/5
  tiempos (s): preparacion_cpu=8.04 · transferencia_h2d=0.00 · entrenamiento=105.71 · evaluacion=9.46 · total=123.23
SECUENCIAL · cnn_reg_drop035 · corrida 3/5
  tiempos (s): preparacion_cpu=7.96 · transferencia_h2d=0.00 · entrenamiento=94.08 · evaluacion=8.17 · total=110.22
SECUENCIAL · cnn_reg_drop035 · corrida 4/5
  tiempos (s): preparacion_cpu=7.93 · transferencia_h2d=0.00 · entrenamiento=98.22 · evaluacion=8.32 · total=114.49
SECUENCIAL · cnn_reg_drop035 · corrida 5/5
  tiempos (s): preparacion_cpu=7.83 · transferencia_h2d=0.00 · entrenamiento=94.19 · evaluacion=8.26 · total=110.29
PARALELA · cnn_reg_drop035 · calentamiento
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
```

Log completo: [`results/logs/benchmark_20261108_002048.log`](../results/logs/benchmark_20261108_002048.log). Extracto (detección de hardware, procesos usados y tiempos por corrida):

```text
  "torch_hilos_intraop": 24,
PARALELA · cnn_reg_drop035 · calentamiento
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=3.45 · transferencia_h2d=0.19 · entrenamiento=1.89 · evaluacion=0.17 · total=7.31
PARALELA · cnn_reg_drop035 · corrida 1/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=4.01 · transferencia_h2d=0.04 · entrenamiento=5.46 · evaluacion=0.39 · total=9.93
PARALELA · cnn_reg_drop035 · corrida 2/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=3.58 · transferencia_h2d=0.05 · entrenamiento=5.74 · evaluacion=0.45 · total=9.83
PARALELA · cnn_reg_drop035 · corrida 3/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=4.09 · transferencia_h2d=0.05 · entrenamiento=6.69 · evaluacion=0.48 · total=11.33
PARALELA · cnn_reg_drop035 · corrida 4/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=4.41 · transferencia_h2d=0.04 · entrenamiento=6.11 · evaluacion=0.46 · total=11.05
PARALELA · cnn_reg_drop035 · corrida 5/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  tiempos (s): preparacion_cpu=5.17 · transferencia_h2d=0.05 · entrenamiento=8.45 · evaluacion=0.54 · total=14.27
```

Log completo: [`results/logs/benchmark_20261108_222830.log`](../results/logs/benchmark_20261108_222830.log). Extracto (detección de hardware, procesos usados y tiempos por corrida):

```text
  "torch_hilos_intraop": 24,
SECUENCIAL · cnn_base_aug · calentamiento
  tiempos (s): preparacion_cpu=11.26 · transferencia_h2d=0.00 · entrenamiento=16.53 · evaluacion=2.96 · total=31.94
SECUENCIAL · cnn_base_aug · corrida 1/5
  tiempos (s): preparacion_cpu=8.75 · transferencia_h2d=0.00 · entrenamiento=144.88 · evaluacion=12.04 · total=165.71
SECUENCIAL · cnn_base_aug · corrida 2/5
  tiempos (s): preparacion_cpu=10.08 · transferencia_h2d=0.00 · entrenamiento=208.90 · evaluacion=15.59 · total=234.60
SECUENCIAL · cnn_base_aug · corrida 3/5
  tiempos (s): preparacion_cpu=24.88 · transferencia_h2d=0.00 · entrenamiento=177.67 · evaluacion=14.27 · total=216.98
SECUENCIAL · cnn_base_aug · corrida 4/5
  tiempos (s): preparacion_cpu=13.48 · transferencia_h2d=0.00 · entrenamiento=207.41 · evaluacion=16.04 · total=236.94
SECUENCIAL · cnn_base_aug · corrida 5/5
  tiempos (s): preparacion_cpu=10.44 · transferencia_h2d=0.00 · entrenamiento=178.47 · evaluacion=13.27 · total=202.19
PARALELA · cnn_base_aug · calentamiento
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 8 workers de CPU
  tiempos (s): preparacion_cpu=5.92 · transferencia_h2d=0.25 · entrenamiento=20.48 · evaluacion=0.19 · total=27.03
PARALELA · cnn_base_aug · corrida 1/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 8 workers de CPU
```

Log completo: [`results/logs/benchmark_20261108_225815.log`](../results/logs/benchmark_20261108_225815.log). Extracto (detección de hardware, procesos usados y tiempos por corrida):

```text
  "torch_hilos_intraop": 24,
PARALELA · cnn_base_aug · calentamiento
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=6.86 · transferencia_h2d=0.04 · entrenamiento=8.46 · evaluacion=0.52 · total=17.06
PARALELA · cnn_base_aug · corrida 1/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=4.57 · transferencia_h2d=0.04 · entrenamiento=27.29 · evaluacion=0.47 · total=32.40
PARALELA · cnn_base_aug · corrida 2/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=4.98 · transferencia_h2d=0.04 · entrenamiento=29.60 · evaluacion=1.24 · total=35.91
PARALELA · cnn_base_aug · corrida 3/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=4.86 · transferencia_h2d=0.04 · entrenamiento=32.31 · evaluacion=0.75 · total=37.99
PARALELA · cnn_base_aug · corrida 4/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=5.00 · transferencia_h2d=0.04 · entrenamiento=34.62 · evaluacion=0.84 · total=40.56
PARALELA · cnn_base_aug · corrida 5/5
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
  [paralelo] preparación con 32 procesos de CPU
  [paralelo] augmentation con 4 workers de CPU
  tiempos (s): preparacion_cpu=5.48 · transferencia_h2d=0.04 · entrenamiento=36.50 · evaluacion=1.30 · total=43.37
```

## 4. Salidas de consola guardadas

**nvidia-smi durante el entrenamiento en GPU (evidencia_fases.py)** · [`evidence/nvidia_smi_entrenamiento.txt`](nvidia_smi_entrenamiento.txt)

```text
Sun Nov  8 22:06:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   59C    P0             52W /   55W |     309MiB /   8188MiB |     67%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     30920    C+G   ...on\157.0.4341.42\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     35568      C   ...0_x64__qbz5n2kfra8p0\python3.12.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

**01 nvidia smi inicial** · [`evidence/consola_01_nvidia_smi_inicial.txt`](consola_01_nvidia_smi_inicial.txt)

```text
Sat Nov  7 21:23:47 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   45C    P8              1W /  115W |     327MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     42388      C   ...0_x64__qbz5n2kfra8p0\python3.12.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

**03 evidencia fases** · [`evidence/consola_03_evidencia_fases.txt`](consola_03_evidencia_fases.txt)

```text
##############################################################################
# FASE 1a ÔÇô PREPARACI├ôN SECUENCIAL (lab original, 1 proceso). CAPTURA AHORA: Administrador de tareas ÔåÆ Rendimiento ÔåÆ CPU (clic derecho ÔåÆ Cambiar gr├ífico a ÔåÆ Procesadores l├│gicos)
##############################################################################
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Cach├® seq: 2,000/3,000
Cach├® seq: 3,000/3,000
Fin de la fase 1a.

##############################################################################
# FASE 1b ÔÇô PREPARACI├ôN PARALELA (32 procesos). CAPTURA AHORA: Administrador de tareas ÔåÆ CPU con todos los n├║cleos
##############################################################################
Fin de la fase 1b (9 pasadas completas por 12,180 im├ígenes).

##############################################################################
# FASE 2 ÔÇô ENTRENAMIENTO EN GPU (CUDA). CAPTURA AHORA: Administrador de tareas ÔåÆ GPU (cambiar un gr├ífico a 'Cuda' o 'Compute_0') y la terminal con nvidia-smi
##############################################################################
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
Sat Nov  7 23:24:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   54C    P0             49W /   55W |     317MiB /   8188MiB |     23%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     33808      C   ...0_x64__qbz5n2kfra8p0\python3.12.exe      N/A      |
|    0   N/A  N/A     42388      C   ...0_x64__qbz5n2kfra8p0\python3.12.exe      N/A      |
+-----------------------------------------------------------------------------------------+

  ├®poca 1 ┬À loss 1.3825
  ├®poca 2 ┬À loss 0.5865
  ├®poca 3 ┬À loss 0.4817
  ├®poca 4 ┬À loss 0.3725
  ├®poca 5 ┬À loss 0.2559
  ├®poca 6 ┬À loss 0.2115
  ├®poca 7 ┬À loss 0.0898
  ├®poca 8 ┬À loss 0.2760
  ├®poca 9 ┬À loss 0.5277
  ├®poca 10 ┬À loss 0.2488
  ├®poca 11 ┬À loss 0.0627
  ├®poca 12 ┬À loss 0.3327
  ├®poca 13 ┬À loss 0.5916
  ├®poca 14 ┬À loss 0.0383
  ├®poca 15 ┬À loss 0.0873
  ├®poca 16 ┬À loss 0.0839

##############################################################################
# FIN DEL MODO EVIDENCIA
##############################################################################
```

**04 bench kernel** · [`evidence/consola_04_bench_kernel.txt`](consola_04_bench_kernel.txt)

```text
Datos: (12180, 64, 64, 3) uint8 = 149.7 MB ┬À bloque (16, 16, 1) ┬À grid (4, 4, 12180)
A. CPU NumPy                              503.887 ms ┬▒ 68.476
B. CPU NumPy + H2D float32                809.173 ms ┬▒ 128.615
C. H2D uint8 + kernel Numba                16.055 ms ┬▒ 0.626
D. Kernel Numba (solo c├│mputo)              3.207 ms ┬▒ 0.058
E. PyTorch nativo en GPU                   13.700 ms ┬▒ 0.012
Verificaci├│n: kernel == NumPy (bit a bit) Ô£ö

Resultados en C:\p3\asl-recognition-cuda\results\kernel_benchmark.md
```

**05 evidencia gpu** · [`evidence/consola_05_evidencia_gpu.txt`](consola_05_evidencia_gpu.txt)

```text
##############################################################################
# FASE 1b – PREPARACIÓN PARALELA (32 procesos). CAPTURA AHORA: Administrador de tareas → CPU con todos los núcleos
##############################################################################
Fin de la fase 1b (26 pasadas completas por 12,180 imágenes).

##############################################################################
# FASE 2 – ENTRENAMIENTO EN GPU (CUDA). CAPTURA AHORA: Administrador de tareas → GPU (cambiar un gráfico a 'Cuda' o 'Compute_0') y la terminal con nvidia-smi
##############################################################################
  [paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)
Sun Nov  8 22:06:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   59C    P0             52W /   55W |     309MiB /   8188MiB |     67%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     30920    C+G   ...on\157.0.4341.42\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     35568      C   ...0_x64__qbz5n2kfra8p0\python3.12.exe      N/A      |
+-----------------------------------------------------------------------------------------+

  época 1 · loss 2.2169
  época 2 · loss 1.1219
  época 3 · loss 0.5568
  época 4 · loss 0.6362
  época 5 · loss 0.3510
  época 6 · loss 0.3730
  época 7 · loss 0.4003
  época 8 · loss 0.3701
  época 9 · loss 0.3300
  época 10 · loss 0.1209
  época 11 · loss 0.0793
  época 12 · loss 0.1098
  época 13 · loss 0.3413
  época 14 · loss 0.0796
  época 15 · loss 0.1868
  época 16 · loss 0.3176
  época 17 · loss 0.1441
  época 18 · loss 0.0950
  época 19 · loss 0.0517
  época 20 · loss 0.0436
  época 21 · loss 0.4108
  época 22 · loss 0.0190
  época 23 · loss 0.0308
  época 24 · loss 0.0232
  época 25 · loss 0.1028

##############################################################################
# FIN DEL MODO EVIDENCIA
##############################################################################
```

## 5. Capturas de pantalla de los monitores de recursos

Capturas incluidas: **12 de 12**.

#### 01_nvidia_smi_inicial · Hardware

nvidia-smi antes de empezar: GPU, driver y versión de CUDA reconocidos.

![01_nvidia_smi_inicial.png](capturas/01_nvidia_smi_inicial.png)

#### 02_check_env · Hardware

check_env.py: PyTorch y Numba ven la GPU y el CPU de 32 hilos; dataset listo.

![02_check_env.png](capturas/02_check_env.png)

#### 03_cpu_preparacion_secuencial · FASE 1 · secuencial

Preparación del lab original: un solo proceso, uso total del CPU bajo.

![03_cpu_preparacion_secuencial.png](capturas/03_cpu_preparacion_secuencial.png)

#### 04_cpu_preparacion_paralela · FASE 1 · paralela

Preparación paralela: los 32 procesadores lógicos cargados.

![04_cpu_preparacion_paralela.png](capturas/04_cpu_preparacion_paralela.png)

#### 05_gpu_entrenamiento_taskmgr · FASE 2 · GPU

Entrenamiento: gráfico Cuda alto y memoria dedicada de la GPU ocupada.

![05_gpu_entrenamiento_taskmgr.png](capturas/05_gpu_entrenamiento_taskmgr.png)

#### 06_cpu_durante_gpu · FASE 2 · GPU

CPU relativamente libre mientras la GPU entrena.

![06_cpu_durante_gpu.png](capturas/06_cpu_durante_gpu.png)

#### 07_nvidia_smi_entrenamiento · FASE 2 · GPU

nvidia-smi durante el entrenamiento: GPU-Util alto y python.exe usando la GPU.

![07_nvidia_smi_entrenamiento.png](capturas/07_nvidia_smi_entrenamiento.png)

#### 08_bench_kernel · Kernel CUDA

Kernel Numba propio vs NumPy, verificado bit a bit.

![08_bench_kernel.png](capturas/08_bench_kernel.png)

#### 09_secuencial_entrenamiento_cpu · Benchmark · secuencial

Entrenamiento secuencial en el CPU; la GPU NVIDIA sin uso.

![09_secuencial_entrenamiento_cpu.png](capturas/09_secuencial_entrenamiento_cpu.png)

#### 10_paralela_entrenamiento_gpu · Benchmark · paralela (capturas 10 y 11 unificadas)

Benchmark oficial, versión paralela de cnn_reg_drop035 (corridas 3–5). En una sola imagen: nvidia-smi -l 2 con GPU-Util 98 %, estado P0, 83 W, 527 MiB y python3.12.exe como proceso tipo C (CUDA); el log con 'GPU detectada', 'preparación con 32 procesos de CPU' y los tiempos por fase; y el Administrador de tareas con los 32 procesadores lógicos mostrando una ráfaga periódica por cada corrida (FASE 1 en el CPU) mientras la GPU entrena entre ráfagas (FASE 2).

![10_paralela_entrenamiento_gpu.png](capturas/10_paralela_entrenamiento_gpu.png)

#### 12_paralela_aug_cpu_y_gpu · Benchmark · paralela

cnn_base_aug durante el entrenamiento (épocas 1–5): GPU en estado P0 con python3.12.exe como proceso CUDA (tipo C) y los workers de augmentation repartidos en los núcleos del CPU. El GPU-Util instantáneo es bajo (4 %): con augmentation la GPU ESPERA a que el CPU genere los lotes. En los registros de nvidia-smi el GPU-Util medio al entrenar es ~14–18 % en este modelo contra ~76 % en cnn_reg_drop035 (sin augmentation): el cuello de botella es la preparación en el CPU, lo que explica su menor speedup. Captura tomada en una corrida extra de 12 épocas (no incluida en los promedios) para tener tiempo de capturar.

![12_paralela_aug_cpu_y_gpu.png](capturas/12_paralela_aug_cpu_y_gpu.png)

#### 13_resumen_speedup · Resultado

Tabla RESUMEN de speedup al final del benchmark.

![13_resumen_speedup.png](capturas/13_resumen_speedup.png)

#### 03b_terminal_fase1a · adicional

![03b_terminal_fase1a.png](capturas/03b_terminal_fase1a.png)

#### 04b_terminal_fase1b · adicional

![04b_terminal_fase1b.png](capturas/04b_terminal_fase1b.png)

#### 13b_resumen_speedup_continuacion · adicional

![13b_resumen_speedup_continuacion.png](capturas/13b_resumen_speedup_continuacion.png)

## 6. Figuras generadas a partir de los logs

**Tiempo medio por fase, secuencial vs paralela.**

![01_tiempos_por_fase.png](../results/figuras/01_tiempos_por_fase.png)

**Speedup por fase (t_secuencial / t_paralela).**

![02_speedup_por_fase.png](../results/figuras/02_speedup_por_fase.png)

**Uso medio de cada núcleo durante la FASE 1 (secuencial vs paralela).**

![03_cpu_preparacion_por_nucleo.png](../results/figuras/03_cpu_preparacion_por_nucleo.png)

**Mapa de calor del uso por núcleo a lo largo de la versión paralela.**

![03_cpu_nucleos_paralela.png](../results/figuras/03_cpu_nucleos_paralela.png)

**Utilización y memoria de la GPU (nvidia-smi) durante todo el benchmark.**

![04_gpu_uso.png](../results/figuras/04_gpu_uso.png)

**Kernel CUDA propio vs NumPy.**

![05_kernel.png](../results/figuras/05_kernel.png)

## 7. Tiempos y speedup (resultado del benchmark)

Fuente: `tiempos.csv` · media ± desviación estándar en segundos · speedup = t_secuencial / t_paralela

## cnn_base_aug

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | 13.53 ± 6.58 | 4.98 ± 0.33 | 2.72× |
| Transferencia H2D | 0.00 ± 0.00 | 0.04 ± 0.00 | – |
| Entrenamiento | 183.47 ± 26.30 | 32.06 ± 3.71 | 5.72× |
| Evaluación | 14.24 ± 1.65 | 0.92 ± 0.35 | 15.50× |
| Total | 211.28 ± 29.13 | 38.04 ± 4.22 | **5.55×** |
| ↳ Carga de lotes (CPU) (incluida en entrenamiento) | 73.88 ± 6.70 | 0.00 ± 0.00 | – |
| ↳ Kernel normalización (incluida en entrenamiento) | 0.00 ± 0.00 | 1.74 ± 0.26 | – |
| ↳ H2D en el bucle (incluida en entrenamiento) | 0.00 ± 0.00 | 0.70 ± 0.11 | – |

Calidad del modelo (promedio de las corridas, conjunto de prueba):

| Versión | Accuracy | Macro F1 |
|---|---:|---:|
| paralela | 0.7329 | 0.7300 |
| secuencial | 0.7559 | 0.7530 |

## cnn_reg_drop035

| Fase | Secuencial (s) | Paralela (s) | Speedup |
|---|---:|---:|---:|
| Preparación (CPU) | 8.14 ± 0.44 | 4.25 ± 0.59 | 1.91× |
| Transferencia H2D | 0.00 ± 0.00 | 0.05 ± 0.00 | – |
| Entrenamiento | 100.76 ± 7.69 | 6.49 ± 1.19 | 15.52× |
| Evaluación | 8.61 ± 0.54 | 0.46 ± 0.05 | 18.60× |
| Total | 117.53 ± 8.49 | 11.28 ± 1.80 | **10.42×** |
| ↳ Carga de lotes (CPU) (incluida en entrenamiento) | 5.72 ± 0.33 | 0.00 ± 0.00 | – |
| ↳ Kernel normalización (incluida en entrenamiento) | 0.00 ± 0.00 | 0.21 ± 0.12 | – |
| ↳ H2D en el bucle (incluida en entrenamiento) | 0.00 ± 0.00 | 0.00 ± 0.00 | – |

Calidad del modelo (promedio de las corridas, conjunto de prueba):

| Versión | Accuracy | Macro F1 |
|---|---:|---:|
| paralela | 0.9455 | 0.9451 |
| secuencial | 0.9130 | 0.9118 |

### Kernel CUDA propio (Numba) vs NumPy

# Kernel de normalización: 12180 imágenes 64×64, 20 repeticiones

| Variante | Media (ms) | SD (ms) | Speedup vs CPU |
|---|---:|---:|---:|
| A. CPU NumPy | 503.887 | 68.476 | 1.00× |
| B. CPU NumPy + H2D float32 | 809.173 | 128.615 | 0.62× |
| C. H2D uint8 + kernel Numba | 16.055 | 0.626 | 31.39× |
| D. Kernel Numba (solo cómputo) | 3.207 | 0.058 | 157.11× |
| E. PyTorch nativo en GPU | 13.700 | 0.012 | 36.78× |
