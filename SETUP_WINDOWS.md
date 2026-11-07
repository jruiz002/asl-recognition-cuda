# Instalación en la laptop Windows (GPU NVIDIA)

Tiempo estimado: **10–15 min** (la mayor parte es la descarga de PyTorch con CUDA, ~2,5 GB).
Todos los comandos son para **PowerShell**, desde la carpeta del repositorio.

## 0. Antes de empezar (si se puede, hacerlo el día anterior)

- [ ] **Driver NVIDIA actualizado.** Abrir PowerShell y correr `nvidia-smi`. La versión del driver
      (arriba a la izquierda) debe ser **≥ 570** para CUDA 12.8. Si es menor, actualizar con GeForce Experience /
      NVIDIA App o desde nvidia.com/drivers.
- [ ] **Python 3.12** instalado (`py -3.12 --version`). Si no está:
      `winget install -e --id Python.Python.3.12` (o el instalador de python.org marcando *Add to PATH*).
- [ ] **Git** instalado (`git --version`). Si no está: `winget install -e --id Git.Git`.
- [ ] Laptop **conectada al cargador** y en modo de energía **Máximo rendimiento**
      (Configuración → Sistema → Energía → Modo de energía).

## 1. Código y datos

```powershell
git clone <URL-del-repo> asl-recognition-cuda
cd asl-recognition-cuda
```

Copiar el dataset desde la USB (`asl_subset.zip`, 225 MB, solo las 17 400 imágenes que se usan):

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
Expand-Archive -Path E:\asl_subset.zip -DestinationPath data\raw   # cambiar E: por la letra de la USB
```

Debe quedar `data\raw\asl_alphabet_train\asl_alphabet_train\A\A1055.jpg`, etc.

> Sin USB: `python -c "from src.data import download_dataset; download_dataset('data/raw')"`
> (descarga el dataset completo de Kaggle, ~1 GB; hacerlo **después** del paso 2).

## 2. Entorno virtual e instalación (EL ORDEN IMPORTA)

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # permite activar el venv en esta ventana
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) PyTorch CON CUDA, desde el índice oficial (NO desde PyPI, que trae la versión solo CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) El resto (Numba con target CUDA de NVIDIA, psutil, pandas, ...)
pip install -r requirements-windows.txt
```

> Si `pip` intentara reinstalar torch en el paso 2, cancelar con Ctrl+C y revisar que el paso 1 terminó bien.
> Si el comando de pytorch.org cambió, el selector de https://pytorch.org/get-started/locally/
> (Stable · Windows · Pip · Python · CUDA 12.8) da el comando actualizado.

## 3. Verificación (primer captura de evidencia)

```powershell
nvidia-smi
python scripts\check_env.py
```

`check_env.py` debe terminar con **`TODO LISTO para el benchmark.`** y deja la salida en `evidence\check_env.txt`.

| Línea | Qué significa si falla | Solución |
|---|---|---|
| `[FALLA] PyTorch NO ve una GPU CUDA` | Se instaló torch solo-CPU | `pip uninstall -y torch` y repetir el paso 2.1 |
| `[FALLA] Numba no encuentra la GPU` / error NVVM | Falta el target CUDA de Numba | `pip install "numba-cuda[cu12]"` |
| `[FALLA] Dataset` | Ruta incorrecta | Revisar que exista `data\raw\asl_alphabet_train\asl_alphabet_train\A` |
| `nvidia-smi` no reconocido | Driver no instalado o no está en el PATH | Instalar o actualizar el driver NVIDIA |

Pruebas automáticas (opcional, ~10 s; incluye la prueba del kernel en la GPU real):

```powershell
python -m unittest discover -s tests -v
```

## 4. Siguiente paso

Seguir **`CHECKLIST_LAPTOP.md`**.

## Notas

- **Administrador de tareas y CUDA:** el gráfico "3D" de la GPU **no** muestra el uso de CUDA. Hay que hacer clic
  en el título de uno de los gráficos pequeños y cambiarlo a **"Cuda"** o **"Compute_0"**.
- Si la laptop tiene GPU integrada Intel + NVIDIA, CUDA siempre usa la NVIDIA; no hace falta configurar nada.
- Para ver todos los núcleos del CPU: Administrador de tareas → Rendimiento → CPU → clic derecho en el
  gráfico → *Cambiar gráfico a* → **Procesadores lógicos**.
