"""Pruebas del Proyecto 3: equivalencia entre la versión secuencial y la paralela.

Ejecutar:
    python -m unittest tests.test_proyecto3 -v
    NUMBA_ENABLE_CUDASIM=1 python -m unittest tests.test_proyecto3 -v   # kernel en simulador (Mac)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from proyecto3.common.paths import load_portable_manifest, split_manifests, stratified_sample
from proyecto3.parallel.cpu_prep import prepare_splits_parallel
from proyecto3.parallel.gpu_kernels import (
    blocks_per_grid,
    launch_normalize,
    normalize_numpy,
    normalize_torch_batch,
    numba_cuda_available,
)
from proyecto3.parallel.gpu_train import epoch_permutation
from src.neural import build_image_cache

LABELS = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "del", "nothing", "space"]


def make_training_tree(root: Path, per_class: int = 3) -> Path:
    """Dataset sintético con la misma estructura que el de Kaggle (JPEG con ruido)."""
    training = root / "asl_alphabet_train" / "asl_alphabet_train"
    generator = np.random.default_rng(0)
    for label in LABELS:
        (training / label).mkdir(parents=True)
        for index in range(per_class):
            pixels = generator.integers(0, 256, (200, 200, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(training / label / f"{label}{index}.jpg")
    return training


class PortableManifestTest(unittest.TestCase):
    def test_windows_paths_are_remapped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_training_tree(root / "raw")
            rows = [
                {"path": f"C:\\Users\\x\\data\\raw\\asl_alphabet_train\\asl_alphabet_train\\{label}\\{label}{i}.jpg",
                 "label": label, "split": split}
                for label in LABELS
                for i, split in enumerate(("train", "validation", "test"))
            ]
            pd.DataFrame(rows).to_csv(root / "all.csv", index=False)
            data = load_portable_manifest(root / "all.csv", root / "raw")
            self.assertTrue(all(Path(path).exists() for path in data["path"]))
            manifests, classes = split_manifests(stratified_sample(data, 1, 42))
            self.assertEqual(len(classes), 29)
            self.assertEqual({len(frame) for frame in manifests.values()}, {29})


class ParallelPreparationTest(unittest.TestCase):
    def test_parallel_cache_is_identical_to_original(self):
        """La FASE 1 paralela produce exactamente los mismos bytes que build_image_cache."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training = make_training_tree(root, per_class=5)
            paths = sorted(str(path) for path in training.rglob("*.jpg"))
            manifest = pd.DataFrame({"path": paths})
            original = np.load(build_image_cache(manifest, root / "original.npy"))
            parallel = np.load(prepare_splits_parallel({"train": paths}, root / "par", 64, workers=3)["train"])
            np.testing.assert_array_equal(original, parallel)


class BatchOrderTest(unittest.TestCase):
    def test_same_order_as_lab_dataloader(self):
        """La versión GPU recorre los lotes en el mismo orden que el DataLoader del lab."""
        # Varios lotes por época y épocas COMPLETAS (el sampler consume el generador
        # también al agotarse; ver epoch_permutation).
        reference = DataLoader(list(range(300)), batch_size=64, shuffle=True,
                               generator=torch.Generator().manual_seed(42))
        mine = torch.Generator().manual_seed(42)
        for _ in range(4):
            lab_order = torch.cat(list(reference)).numpy()
            np.testing.assert_array_equal(lab_order, epoch_permutation(mine, 300).numpy())


class KernelTest(unittest.TestCase):
    def setUp(self):
        self.batch = np.random.default_rng(1).integers(0, 256, (4, 21, 19, 3), dtype=np.uint8)
        # Exactamente lo que hacía ASLImageDataset: float32 / 255 y transpose(2, 0, 1).
        self.reference = np.stack([
            (image.astype(np.float32) / 255.0).transpose(2, 0, 1) for image in self.batch
        ])

    def test_grid_covers_every_pixel(self):
        self.assertEqual(blocks_per_grid(64, 64, 64), (4, 4, 64))
        self.assertEqual(blocks_per_grid(4, 21, 19), (2, 2, 4))

    def test_numpy_reference(self):
        np.testing.assert_array_equal(normalize_numpy(self.batch), self.reference)

    def test_torch_cpu_fallback(self):
        result = normalize_torch_batch(torch.from_numpy(self.batch)).numpy()
        np.testing.assert_array_equal(result, self.reference)

    @unittest.skipUnless(numba_cuda_available(), "requiere GPU o NUMBA_ENABLE_CUDASIM=1")
    def test_numba_kernel_matches_cpu(self):
        from numba import cuda

        source = cuda.to_device(self.batch)
        output = cuda.device_array((4, 3, 21, 19), dtype=np.float32)
        launch_normalize(source, output)
        np.testing.assert_array_equal(output.copy_to_host(), self.reference)

    @unittest.skipUnless(torch.cuda.is_available(), "requiere GPU NVIDIA real")
    def test_numba_kernel_with_torch_tensors(self):
        result = normalize_torch_batch(torch.from_numpy(self.batch).cuda()).cpu().numpy()
        np.testing.assert_array_equal(result, self.reference)


if __name__ == "__main__":
    unittest.main()
