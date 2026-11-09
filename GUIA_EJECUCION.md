# Guía de ejecución y de capturas – Proyecto 3 (ASL + CUDA)

Guía práctica para la laptop Windows (i9-13980HX + RTX 4060 Laptop). Dice **qué comando correr**,
**qué prueba cada uno** (CPU, GPU o ambos) y **en qué momento tomar cada captura**.

> Carpeta del proyecto en la laptop: **`C:\p3\asl-recognition-cuda`**. No uses la copia de OneDrive:
> OneDrive sincroniza la caché y el `.venv` mientras se mide, y eso ensucia los tiempos.

---

## 0. Preparar la sesión (cada vez que abras una terminal)

```powershell
cd C:\p3\asl-recognition-cuda
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # permite activar el venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"                               # tildes y símbolos en la consola
$env:PYTHONUNBUFFERED = "1"                                   # la salida aparece al instante aunque se use Tee-Object
mkdir evidence\capturas -Force | Out-Null                     # carpeta de las capturas
```

Antes de medir:
- Laptop **conectada al cargador** y modo de energía en **Máximo rendimiento**.
- **OneDrive en pausa**. Cerrar el navegador, Teams y otros programas pesados (la laptop tiene 16 GB de RAM).

Disposición de ventanas recomendada:

| Ventana | Para qué |
|---|---|
| **Terminal A** | Comandos principales (`benchmark.py`, scripts) |
| **Terminal B** | `nvidia-smi -l 2` (monitor de la GPU) |
| **Administrador de tareas** (Ctrl+Shift+Esc → *Rendimiento*) | Capturas de CPU y GPU |

Configuración del Administrador de tareas (una sola vez):
- **CPU** → clic derecho en el gráfico → *Cambiar gráfico a* → **Procesadores lógicos** (se ven los 32 hilos).
- **GPU 1 (NVIDIA GeForce RTX 4060)** → clic en el título de uno de los gráficos pequeños (dice "3D",
  "Copy", etc.) → elegir **Cuda** (o **Compute_0**). El gráfico "3D" **no** muestra el trabajo de CUDA.

Las capturas se toman con **Win + Shift + S** y se pegan en Paint para guardarlas como PNG en `evidence\capturas\`
con el nombre exacto de esta guía.

---

## 1. Mapa rápido: qué comando prueba qué

| Objetivo | Comando | Dónde corre |
|---|---|---|
| Verificar hardware y librerías | `python scripts\check_env.py` | CPU + GPU (diagnóstico) |
| Tests automáticos | `python -m unittest discover -s tests -v` | CPU + GPU (incluye el kernel en la GPU real) |
| **Solo la versión secuencial (CPU)** | `python benchmark.py --only sequential` | **CPU**: lab original, 1 proceso para los datos |
| **Solo la versión paralela (CPU multinúcleo + GPU)** | `python benchmark.py --only parallel` | FASE 1 en **CPU multinúcleo**, FASE 2 en **GPU** |
| Versión paralela **sin GPU** (prueba de respaldo y de corrección) | `$env:CUDA_VISIBLE_DEVICES="-1"; python benchmark.py --only parallel --sample 100 --epochs 2 --runs 1 --no-warmup --no-plots --experiments cnn_reg_drop035` y después `$env:CUDA_VISIBLE_DEVICES=$null` | Todo en CPU; imprime `AVISO: no hay GPU CUDA`, no falla y da **la misma pérdida y el mismo F1** que la secuencial (prueba de que la lógica es idéntica). Ojo: `""` no sirve en PowerShell porque borra la variable |
| **Armar el documento único de evidencia** | `python scripts\armar_evidencia.py` | — (lee logs, CSV y capturas → `evidence\EVIDENCIA.md`) |
| Solo el kernel CUDA propio (Numba) contra NumPy | `python scripts\bench_kernel.py --reps 20` | GPU (kernel) vs CPU (NumPy) |
| Mantener cada fase activa para las capturas | `python scripts\evidencia_fases.py --seconds 40` | CPU 1 núcleo → CPU todos → GPU |
| **Benchmark oficial (las dos versiones)** | `python benchmark.py --fresh` | Ambas, mismos datos, N corridas |
| Regenerar gráficas y tabla desde el CSV | `python scripts\make_plots.py` | — |

Flags útiles de `benchmark.py`:

| Flag | Efecto |
|---|---|
| `--runs N` | Corridas medidas por versión (5 por defecto; más 1 de calentamiento que no se cuenta) |
| `--epochs N` | Épocas por corrida (6, igual que el lab) |
| `--sample N` | Imágenes por clase y partición (modo prueba rápida) |
| `--only {sequential,parallel,both}` | Qué versión correr |
| `--experiments cnn_reg_drop035 cnn_base_aug` | Qué modelos correr |
| `--threads N` | Hilos de PyTorch (con `--threads 1` el baseline es estrictamente de 1 hilo) |
| `--workers N` | Procesos de CPU para la preparación paralela (por defecto, los 32) |
| `--fresh` | Borra `results\tiempos.csv` antes de empezar (resultados limpios) |
| `--no-warmup`, `--no-plots` | Omite el calentamiento o las gráficas |

---

## 2. Secuencia completa con las capturas

### Paso 1 · Evidencia del hardware (≈ 2 min)

**Terminal A:**
```powershell
nvidia-smi
```
📸 **`01_nvidia_smi_inicial.png`**: la terminal con la tabla de `nvidia-smi` (RTX 4060, driver 555.97, CUDA 12.5, 0 % de uso).

```powershell
nvidia-smi | Tee-Object evidence\consola_01_nvidia_smi_inicial.txt
python scripts\check_env.py
```
(`Tee-Object` muestra la salida en pantalla **y** la guarda como log para el documento de evidencia).

📸 **`02_check_env.png`**: las 7 líneas `[OK]` y `TODO LISTO para el benchmark.`
(Además queda guardado como texto en `evidence\check_env.txt`).

### Paso 2 · Prueba rápida (≈ 1–2 min, ya hecha una vez)
```powershell
python benchmark.py --sample 20 --epochs 1 --runs 1 --no-plots
```
Debe decir `[paralelo] GPU detectada: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.6)`. Sus tiempos **no**
sirven: con tan pocas imágenes domina el arranque de procesos. El paso 5 los borra con `--fresh`.

### Paso 3 · Capturas fase por fase (≈ 3 min)

El benchmark real prepara las imágenes en pocos segundos, demasiado rápido para capturarlo. Este script mantiene
**cada fase activa 40 s** y avisa con un bloque `#####` cuándo capturar.

**Terminal A:**
```powershell
python scripts\evidencia_fases.py --seconds 40 | Tee-Object evidence\consola_03_evidencia_fases.txt
```

| Cuando aparezca en la Terminal A… | Ventana | Captura | Qué debe verse |
|---|---|---|---|
| `FASE 1a – PREPARACIÓN SECUENCIAL` | Admin. de tareas → **CPU** (procesadores lógicos) | 📸 **`03_cpu_preparacion_secuencial.png`** | Uso total bajo (≈ 1 de 32 hilos). Windows mueve ese único proceso entre núcleos, por eso se ven varias gráficas con carga baja |
| `FASE 1b – PREPARACIÓN PARALELA` | Admin. de tareas → **CPU** | 📸 **`04_cpu_preparacion_paralela.png`** | **Los 32 hilos** muy cargados, uso total cerca del 100 % |
| `FASE 2 – ENTRENAMIENTO EN GPU` | Admin. de tareas → **GPU 1 (NVIDIA)** | 📸 **`05_gpu_entrenamiento_taskmgr.png`** | Gráfico **Cuda** alto y **memoria dedicada** de la GPU ocupada |
| `FASE 2 – ENTRENAMIENTO EN GPU` (misma fase, cambiar de pestaña) | Admin. de tareas → **CPU** | 📸 **`06_cpu_durante_gpu.png`** | CPU relativamente libre mientras la GPU calcula |
| Al terminar, el script imprime la salida de `nvidia-smi` | **Terminal A** | 📸 **`07_nvidia_smi_entrenamiento.png`** | `GPU-Util` alto y el proceso `python.exe` usando memoria de la GPU |

Además se guarda automáticamente `evidence\nvidia_smi_entrenamiento.txt`.

### Paso 4 · Kernel CUDA propio vs NumPy (≈ 1 min)

**Terminal A:**
```powershell
python scripts\bench_kernel.py --reps 20 | Tee-Object evidence\consola_04_bench_kernel.txt
```
📸 **`08_bench_kernel.png`**: la tabla de tiempos (NumPy en CPU vs kernel Numba en GPU, con y sin transferencias) y
la línea `Verificación: kernel == NumPy (bit a bit) ✔`.

### Paso 5 · BENCHMARK OFICIAL (≈ 30–45 min con 5 corridas)

**Terminal B** (dejarla corriendo todo el benchmark):
```powershell
nvidia-smi -l 2
```

**Terminal A:**
```powershell
python benchmark.py --fresh --loader-workers 4    # si hay poco tiempo: agregar --runs 3
```

> **`--loader-workers 4` (laptop con 16 GB de RAM):** en Windows cada worker del DataLoader es un proceso que
> importa torch (~0,9 GB de memoria comprometida cada uno). Con los 8 por defecto, la RAM libre bajó a 0,75 GB y un
> worker murió en `cnn_base_aug` (`WinError 6 Controlador no válido` / `DataLoader worker exited unexpectedly`).
> Con 4 workers el benchmark terminó sin errores.
>
> **Si el benchmark se interrumpe**, no hace falta repetir lo que ya terminó: las corridas completas quedan en
> `results\tiempos.csv`. Se continúa **sin `--fresh`** con `--only` y `--experiments` (por ejemplo
> `python benchmark.py --only parallel --experiments cnn_base_aug --loader-workers 4`), y
> `armar_evidencia.py` junta todas las ejecuciones e ignora las corridas interrumpidas.

Orden de ejecución (cada bloque se anuncia con una línea `=====` en la Terminal A):
1. `SECUENCIAL · cnn_reg_drop035`: calentamiento y N corridas
2. `PARALELA · cnn_reg_drop035`: calentamiento y N corridas
3. `SECUENCIAL · cnn_base_aug`: calentamiento y N corridas
4. `PARALELA · cnn_base_aug`: calentamiento y N corridas
5. `RESUMEN`: tabla de speedup y gráficas

**No usar la laptop durante el benchmark**, salvo para las capturas (son livianas).

| Cuando en la Terminal A esté… | Ventana | Captura | Qué debe verse |
|---|---|---|---|
| `SECUENCIAL · cnn_reg_drop035 · corrida 1`, mientras imprime `epoch=…` | Admin. de tareas → **CPU** (y de reojo GPU 1 en ~0 %) | 📸 **`09_secuencial_entrenamiento_cpu.png`** | Hasta 8 hilos ocupados por PyTorch en el CPU; la GPU NVIDIA sin uso |
| `PARALELA · cnn_reg_drop035 · corrida 1`, mientras imprime `epoch=…` | Admin. de tareas → **GPU 1** (Cuda) | 📸 **`10_paralela_entrenamiento_gpu.png`** | Cuda alto y memoria dedicada ocupada |
| La misma fase | **Terminal B** (`nvidia-smi -l 2`) | 📸 **`11_nvidia_smi_paralela.png`** | `GPU-Util` alto y `python.exe` en la lista de procesos |
| `PARALELA · cnn_base_aug · corrida 1`, mientras imprime `epoch=…` | Admin. de tareas: pestaña **CPU** y luego **GPU 1** (o las dos ventanas lado a lado) | 📸 **`12_paralela_aug_cpu_y_gpu.png`** | Los workers de augmentation cargando el CPU **y** la GPU calculando al mismo tiempo |
| `RESUMEN` al final | **Terminal A** | 📸 **`13_resumen_speedup.png`** | La tabla de speedup por fase |

Archivos que el benchmark genera solo (también cuentan como evidencia y logs):

| Archivo | Contenido |
|---|---|
| `results\tiempos.csv` | Una fila por corrida con los segundos por fase y las métricas |
| `results\resumen.csv` | Media y desviación estándar por versión, modelo y fase |
| `results\tabla_speedup.md` | Tabla de tiempos y speedup lista para el informe |
| `results\hw_info.json` | CPU, GPU, driver y versiones de las librerías |
| `results\logs\benchmark_*.log` | Toda la salida de la consola |
| `results\figuras\*.png` | Gráficas de tiempos por fase y de speedup |
| `evidence\nvidia_smi_*.csv` | Uso de la GPU cada segundo durante cada versión |
| `evidence\cpu_por_nucleo_*.csv` | Uso de **cada núcleo** del CPU, etiquetado por fase |

### Paso 6 · (Opcional) baseline estrictamente de 1 hilo (≈ 15 min)
```powershell
python benchmark.py --only sequential --threads 1 --runs 2 --experiments cnn_reg_drop035 --no-plots
```
Agrega filas con la versión `secuencial_1hilos`, que no se mezclan con el baseline. Sirven para discutir
cuánto aportan por sí solos los hilos de PyTorch en el CPU.

### Paso 7 · Armar el documento único de evidencia (≈ 10 s)

Con todas las capturas ya guardadas en `evidence\capturas\`:
```powershell
python scripts\armar_evidencia.py
code evidence\EVIDENCIA.md        # abrir y ver con Ctrl+Shift+V (vista previa con imágenes)
```
`evidence\EVIDENCIA.md` reúne en un solo documento: hardware detectado, **tabla de uso de hardware por fase**
(CPU total, núcleos activos y GPU-Util de cada fase, sacados de los CSV de monitoreo), extractos del log, salidas de
consola, las 13 capturas con su explicación (avisa cuáles faltan) y las figuras. Si se agrega una captura después,
basta con volver a correr el comando.

### Paso 8 · Guardar los resultados
```powershell
git status
git add results evidence requirements-windows.txt GUIA_EJECUCION.md scripts\armar_evidencia.py
git commit -m "Resultados del benchmark en laptop (RTX 4060 + i9-13980HX)"
git push
```
Si la laptop no tiene credenciales de GitHub, comprimir `results\` y `evidence\` en un zip y copiarlo a la USB.

---

## 3. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `UnicodeEncodeError` | `$env:PYTHONIOENCODING="utf-8"` |
| `PermissionError` al borrar `data\bench_cache` | Cerrar las terminales de Python, borrar `data\bench_cache` a mano y repetir |
| `BrokenProcessPool` | Correr siempre como script (`python benchmark.py`), nunca pegando código en la consola |
| `nvJitLinkError` / `np.row_stack` | `pip install -r requirements-windows.txt` (fija NumPy < 2.5 y las piezas CUDA de Numba a 12.6) |
| La GPU no aparece en el Admin. de tareas con carga | Revisar que el gráfico esté en **Cuda**, no en "3D" |
