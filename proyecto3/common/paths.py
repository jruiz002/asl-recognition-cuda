"""Carga de la partición original con rutas portables (Mac / Windows).

Los manifiestos del laboratorio guardan rutas ABSOLUTAS de la máquina donde se
generaron (``C:\\Users\\dilar\\...\\asl_alphabet_train\\A\\A1055.jpg``). Para reutilizar
exactamente la misma partición en cualquier equipo se reconstruye cada ruta como
``<carpeta de entrenamiento local>/<etiqueta>/<archivo>``.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pandas as pd

from src.data import find_training_directory


def _file_name(raw_path: str) -> str:
    """Obtiene el nombre del archivo tanto de rutas Windows como POSIX."""
    return PureWindowsPath(raw_path).name if "\\" in raw_path else Path(raw_path).name


def load_portable_manifest(manifest_path: Path, data_root: Path) -> pd.DataFrame:
    """Lee ``all.csv`` y reescribe la columna ``path`` hacia el dataset local."""
    data = pd.read_csv(manifest_path)
    required = {"path", "label", "split"}
    if not required.issubset(data.columns):
        raise ValueError(f"El manifiesto debe contener {sorted(required)}")

    training_dir = find_training_directory(data_root)
    data["path"] = [
        str(training_dir / str(label) / _file_name(str(raw)))
        for raw, label in zip(data["path"], data["label"])
    ]
    missing = [path for path in data["path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {len(missing)} imágenes del manifiesto; primera: {missing[0]}"
        )
    return data


def stratified_sample(data: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    """Modo de prueba rápida: toma ``per_class`` imágenes por clase y partición.

    Se mantiene la proporción de particiones y todas las clases, para que el flujo
    completo (train/validation/test) se ejecute igual que con los datos completos.
    """
    if per_class is None:
        return data
    return data.groupby(["split", "label"]).head(per_class).reset_index(drop=True)


def split_manifests(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Separa en train/validation/test con el mismo orden que ``entrenar_modelos.load_manifests``."""
    manifests = {
        split: frame.reset_index(drop=True) for split, frame in data.groupby("split")
    }
    if set(manifests) != {"train", "validation", "test"}:
        raise ValueError("Se requieren train, validation y test")
    return manifests, sorted(data["label"].unique())
