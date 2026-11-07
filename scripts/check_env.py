"""Verificación del entorno ANTES del benchmark (primer comando en la laptop).

Comprueba: versiones, CPU/núcleos, GPU visible para PyTorch, una operación en CUDA,
compilación y ejecución del kernel Numba, dataset y manifiesto. Guarda la salida en
evidence/check_env.txt como evidencia de que el entorno reconoció el hardware.

Uso:  python scripts/check_env.py
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "[OK]  ", "[FALLA]", "[AVISO]"


def run_checks() -> bool:
    import numpy as np
    import torch

    from proyecto3.common.hw_info import collect_hw_info

    healthy = True
    info = collect_hw_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print()
    print(f"{OK} CPU: {info['cpu']} · {info['nucleos_fisicos']} núcleos físicos / {info['nucleos_logicos']} lógicos")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        a = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize()
        value = float((a @ a).sum())
        print(f"{OK} PyTorch ve la GPU: {torch.cuda.get_device_name(0)} · CUDA {torch.version.cuda} "
              f"· cuDNN {torch.backends.cudnn.version()} · matmul de prueba = {value:.1f}")
    else:
        healthy = False
        print(f"{FAIL} PyTorch NO ve una GPU CUDA. ¿Se instaló la versión CPU de torch? "
              "Reinstalar con el comando de SETUP_WINDOWS.md.")

    try:
        from proyecto3.parallel.gpu_kernels import launch_normalize, normalize_numpy, numba_cuda_available
        if numba_cuda_available():
            from numba import cuda

            batch = np.random.default_rng(0).integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
            output = cuda.device_array((8, 3, 64, 64), dtype=np.float32)
            launch_normalize(cuda.to_device(batch), output)
            np.testing.assert_array_equal(output.copy_to_host(), normalize_numpy(batch))
            print(f"{OK} Kernel Numba compilado y verificado (== NumPy) en {cuda.get_current_device().name}")
            if torch.cuda.is_available():
                from proyecto3.parallel.gpu_kernels import normalize_torch_batch

                tensor = torch.from_numpy(batch).cuda()
                np.testing.assert_array_equal(normalize_torch_batch(tensor).cpu().numpy(), normalize_numpy(batch))
                print(f"{OK} Interoperabilidad Numba ↔ PyTorch (zero-copy) verificada")
        else:
            healthy = False
            print(f"{FAIL} Numba no encuentra la GPU. Instalar: pip install \"numba-cuda[cu12]\"")
    except Exception as error:  # noqa: BLE001
        healthy = False
        print(f"{FAIL} Error con el kernel Numba: {type(error).__name__}: {error}")

    try:
        from proyecto3.common.paths import load_portable_manifest

        data = load_portable_manifest(ROOT / "data" / "manifests" / "all.csv", ROOT / "data" / "raw")
        print(f"{OK} Dataset y manifiesto: {len(data):,} imágenes encontradas "
              f"({data.groupby('split').size().to_dict()})")
    except Exception as error:  # noqa: BLE001
        healthy = False
        print(f"{FAIL} Dataset: {error}. Copiar el dataset a data/raw (ver SETUP_WINDOWS.md).")

    try:
        import psutil  # noqa: F401
        print(f"{OK} psutil disponible (monitor de núcleos)")
    except ImportError:
        healthy = False
        print(f"{FAIL} Falta psutil: pip install psutil")

    import shutil

    print(f"{OK if shutil.which('nvidia-smi') else WARN} nvidia-smi "
          f"{'disponible' if shutil.which('nvidia-smi') else 'no encontrado (no habrá log de GPU)'}")
    print("\n" + ("TODO LISTO para el benchmark." if healthy else "HAY PROBLEMAS: revisar las líneas [FALLA]."))
    return healthy


def main() -> int:
    from proyecto3.common import consola_utf8

    consola_utf8()
    buffer = io.StringIO()

    class Both(io.TextIOBase):
        def write(self, text):
            sys.__stdout__.write(text)
            buffer.write(text)
            return len(text)

    with redirect_stdout(Both()):
        healthy = run_checks()
    output = ROOT / "evidence" / "check_env.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nGuardado en {output}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
