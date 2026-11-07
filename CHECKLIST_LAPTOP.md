# Checklist del día de la laptop

**Tiempo total estimado: ~75 min** (15 de instalación + ~10 de pruebas y evidencia + ~45 de benchmark + 5 de cierre).
Si hay menos tiempo: usar `--runs 3` en el paso 6 (ahorra ~15 min).

Llevar: USB con `asl_subset.zip` · cargador de la laptop · URL del repo.

Guardar todas las capturas en `evidence\capturas\` con el nombre indicado (Win + Shift + S para recortar,
luego pegar en Paint y guardar como PNG).

---

## Paso 1 · Instalación (≈ 15 min)
- [ ] Seguir `SETUP_WINDOWS.md` hasta que `check_env.py` diga **TODO LISTO**.
- [ ] Cerrar navegador, Teams, OneDrive y otros programas pesados. Laptop **conectada al cargador**.

## Paso 2 · Evidencia de hardware (≈ 2 min)
```powershell
nvidia-smi
python scripts\check_env.py
```
- [ ] 📸 `01_nvidia_smi_inicial.png` → terminal con la salida de `nvidia-smi` (GPU, driver y CUDA).
- [ ] 📸 `02_check_env.png` → terminal con las líneas `[OK]` de `check_env.py`.

## Paso 3 · Prueba rápida (≈ 3 min)
```powershell
python benchmark.py --sample 20 --epochs 1 --runs 1 --no-plots
```
- [ ] Debe decir `GPU detectada: NVIDIA ...` en la versión paralela y terminar sin errores.
      (Estos tiempos se descartan: el paso 6 usa `--fresh`.)

## Paso 4 · Capturas por fase (≈ 4 min)

Abrir el **Administrador de tareas** (Ctrl + Shift + Esc) → **Rendimiento**. Preparar:
- **CPU:** clic derecho en el gráfico → *Cambiar gráfico a* → **Procesadores lógicos**.
- **GPU NVIDIA:** cambiar uno de los gráficos pequeños (clic en su título) a **Cuda** o **Compute_0**.

En otra terminal (con el venv activado):
```powershell
python scripts\evidencia_fases.py --seconds 40
```
El script avisa en pantalla con `#####` cuándo capturar:

| Aviso en pantalla | Captura | Qué debe verse |
|---|---|---|
| `FASE 1a – PREPARACIÓN SECUENCIAL` | 📸 `03_cpu_preparacion_secuencial.png` | Uso total bajo (≈ 1 núcleo de N; Windows **reparte/migra** ese único proceso entre núcleos, así que se ven varias gráficas con carga baja) |
| `FASE 1b – PREPARACIÓN PARALELA` | 📸 `04_cpu_preparacion_paralela.png` | **Todos los núcleos** cerca del 100 % |
| `FASE 2 – ENTRENAMIENTO EN GPU` | 📸 `05_gpu_entrenamiento_taskmgr.png` | Gráfico **Cuda/Compute** alto y memoria dedicada de la GPU ocupada |
| `FASE 2 – ENTRENAMIENTO EN GPU` | 📸 `06_cpu_durante_gpu.png` | Pestaña CPU: **CPU relativamente libre** mientras la GPU trabaja |

- [ ] El script guarda solo `evidence\nvidia_smi_entrenamiento.txt` (nvidia-smi durante el entrenamiento).
- [ ] 📸 `07_nvidia_smi_entrenamiento.png` → la terminal mostrando esa salida de nvidia-smi (aparece en pantalla).

## Paso 5 · Speedup del kernel Numba (≈ 1 min)
```powershell
python scripts\bench_kernel.py --reps 20
```
- [ ] Debe terminar con `Verificación: kernel == NumPy (bit a bit) ✔`.
- [ ] 📸 `08_bench_kernel.png` → la tabla de la terminal.

## Paso 6 · BENCHMARK COMPLETO (≈ 45 min con 5 corridas, ≈ 30 min con 3)
```powershell
python benchmark.py --fresh            # o:  python benchmark.py --fresh --runs 3
```
Orden de ejecución: `cnn_reg_drop035` secuencial (calentamiento + N) → paralela (calentamiento + N) →
`cnn_base_aug` secuencial → paralela. Cada bloque está marcado con `=====` en la terminal.

En una **segunda terminal** dejar corriendo `nvidia-smi -l 2` (actualiza cada 2 s).

**No usar la laptop durante el benchmark** (salvo para las capturas: son livianas).

| Momento (ver el título `=====` en la terminal) | Captura |
|---|---|
| `SECUENCIAL · cnn_reg_drop035 · corrida 1` durante las épocas | 📸 `09_secuencial_entrenamiento_cpu.png` → CPU: hasta 8 hilos ocupados por PyTorch; GPU NVIDIA en ~0 % |
| `PARALELA · cnn_reg_drop035 · corrida 1` durante las épocas | 📸 `10_paralela_entrenamiento_gpu.png` → Cuda alto · 📸 `11_nvidia_smi_paralela.png` → terminal `nvidia-smi -l 2` con GPU-Util alto |
| `PARALELA · cnn_base_aug · corrida 1` durante las épocas | 📸 `12_paralela_aug_cpu_y_gpu.png` → CPU con los workers activos **y** GPU trabajando al mismo tiempo |
| Al terminar (`RESUMEN`) | 📸 `13_resumen_speedup.png` → la tabla de speedup de la terminal |

- [ ] Al final deben existir: `results\tiempos.csv`, `results\resumen.csv`, `results\tabla_speedup.md`,
      `results\hw_info.json`, `results\logs\benchmark_*.log`, `results\figuras\*.png`,
      `evidence\nvidia_smi_*.csv`, `evidence\cpu_por_nucleo_*.csv`.
- [ ] Revisar rápido `results\figuras\02_speedup_por_fase.png`.

## Paso 7 · (Opcional, si sobra tiempo ≈ 15 min) baseline estrictamente de 1 hilo
```powershell
python benchmark.py --only sequential --threads 1 --runs 2 --experiments cnn_reg_drop035 --no-plots
```
Agrega filas a `tiempos.csv` con versión `secuencial_1hilos` (no se mezclan con el baseline en los promedios;
quedan en `resumen.csv` para analizarlas aparte).

## Paso 8 · Subir resultados (≈ 3 min)
```powershell
git add results evidence
git commit -m "Resultados del benchmark en laptop (RTX + i9)"
git push
```
Si no hay credenciales de Git en la laptop: comprimir `results\` y `evidence\` y copiarlos a la USB.

- [ ] Desinstalar/borrar si la dueña lo pide: carpeta del repo y `.venv` (no queda nada instalado fuera de esa carpeta,
      salvo Python y Git si se instalaron en el paso 0).
