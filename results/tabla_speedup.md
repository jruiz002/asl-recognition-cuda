# Tiempos por fase y speedup

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
