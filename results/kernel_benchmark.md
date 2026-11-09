# Kernel de normalización: 12180 imágenes 64×64, 20 repeticiones

| Variante | Media (ms) | SD (ms) | Speedup vs CPU |
|---|---:|---:|---:|
| A. CPU NumPy | 503.887 | 68.476 | 1.00× |
| B. CPU NumPy + H2D float32 | 809.173 | 128.615 | 0.62× |
| C. H2D uint8 + kernel Numba | 16.055 | 0.626 | 31.39× |
| D. Kernel Numba (solo cómputo) | 3.207 | 0.058 | 157.11× |
| E. PyTorch nativo en GPU | 13.700 | 0.012 | 36.78× |
