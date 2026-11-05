# Paso 1 – Análisis de la solución secuencial existente

> Base: código del **Laboratorio 3 (CC3084 Data Science)**, copiado sin cambios en este repositorio
> (`analisis_exploratorio.py`, `entrenar_modelos.py`, `evaluar_fotos.py`, `src/`).

## 1.1 Contexto técnico

| Aspecto | Valor en el laboratorio |
|---|---|
| Lenguaje / framework | Python + **PyTorch** (redes), scikit-learn (SVM, métricas), Pillow (imágenes) |
| Dataset | Kaggle *ASL Alphabet*: 87 000 JPEG de 200×200 RGB, 29 clases (A–Z, `del`, `nothing`, `space`) |
| Submuestra usada | 600 imágenes por clase = **17 400** (`src/config.py`) |
| Partición | 70 / 15 / 15 estratificada, semilla 42 → **12 180 train · 2 610 validation · 2 610 test** (`data/manifests/`) |
| Entrada de las redes | RGB 64×64, `float32` en [0, 1], formato CHW |
| Entrenamiento | AdamW (lr según experimento, `weight_decay=1e-4`), CrossEntropy, batch 64, 6 épocas, early stopping (patience 2) por macro F1 de validación |
| Modelos | 2 CNN × 2 hiperparámetros, 2 MLP, 3 versiones con augmentation, SVM-HOG (C = 0.1 / 1.0 y con augmentation) |
| Paralelismo original | Ninguno explícito: un solo proceso lee las imágenes (`num_workers=0`). Solo las operaciones matemáticas de PyTorch usan varios hilos internos (`torch.set_num_threads(min(8, …))`) |

## 1.2 Partes principales del programa

| Parte | Archivo / función | Función que cumple |
|---|---|---|
| Configuración | `src/config.py` → `ProjectConfig` | Fija semilla, tamaño de muestra, resolución, proporciones y batch. |
| Adquisición de datos | `src/data.py` → `download_dataset`, `find_training_directory` | Descarga y extrae el ZIP de Kaggle y localiza la carpeta con las 29 clases. |
| Inventario y partición | `src/data.py` → `build_inventory`, `balanced_subsample`, `stratified_split`, `save_manifests` | Lista todos los archivos, toma 600 por clase y crea la partición train/validation/test en CSV. |
| Análisis exploratorio | `analisis_exploratorio.py`, `src/eda.py` | Distribución de clases, cabeceras, ejemplos, estadísticas visuales y letras parecidas. |
| Preprocesamiento | `src/neural.py` → `build_image_cache`, `ASLImageDataset` | Decodifica el JPEG, convierte a RGB, redimensiona a 64×64, guarda una caché `uint8` y, por cada muestra, aplica augmentation opcional, `/255` y HWC→CHW. |
| Augmentation | `src/neural.py` → `SafeAugmentation` | Rotación ±12°, escala ±8 %, traslación ±6 %, contraste y brillo (sin flip horizontal). |
| Modelos | `src/neural.py` → `CNNBaseline`, `CNNRegularized`, `SimpleMLP` | Definen las arquitecturas de PyTorch. |
| Bucle de entrenamiento | `src/neural.py` → `train_experiment` | Para cada época: forward, pérdida, backward y AdamW por lote; luego valida y aplica early stopping. |
| Evaluación | `src/neural.py` → `predict_loader`, `evaluate_loader` | Inferencia por lotes, softmax/argmax, accuracy y macro F1. |
| Modelo clásico | `src/classical.py` → `hog_descriptor`, `train_linear_svm` | Descriptor HOG hecho a mano en NumPy + `LinearSVC`. |
| Orquestación y reportes | `entrenar_modelos.py` | Corre los 9 experimentos neuronales y los 3 SVM, guarda métricas y gráficas y elige el mejor modelo. |
| Prueba con fotos propias | `evaluar_fotos.py` | Predice las fotos del equipo (`data/external/`) con el mejor modelo. |

## 1.3 Descomposición en operaciones (desde que llega la imagen hasta el resultado)

Flujo de **un experimento neuronal** (`train_experiment`), que es lo que se acelera:

| No. | Operación identificada | Descripción de su funcionamiento | Dónde corre en el lab |
|---|---|---|---|
| 1 | Lectura del manifiesto | `pd.read_csv("all.csv")` y separación en train/validation/test; verificación de que existen los archivos. | CPU, 1 hilo |
| 2 | Lectura de archivo | `Image.open(path)`: lee del disco los bytes del JPEG (~14 KB por imagen). | CPU + disco |
| 3 | Decodificación JPEG | Pillow decodifica (Huffman + IDCT por bloques 8×8) a 200×200×3. | CPU, 1 hilo |
| 4 | Conversión a RGB | `convert("RGB")` garantiza 3 canales. | CPU |
| 5 | Redimensionamiento | `resize((64, 64), BILINEAR)`: interpolación bilineal 200×200 → 64×64. | CPU |
| 6 | Escritura en caché | Copia del arreglo `uint8` 64×64×3 a un `.npy` memmap (`build_image_cache`). | CPU + disco |
| 7 | Muestreo de lote | `DataLoader(shuffle=True)` genera la permutación y toma 64 índices. | CPU |
| 8 | Lectura desde caché | `Image.fromarray(cache[i])` para cada muestra del lote. | CPU |
| 9 | Augmentation (si aplica) | `SafeAugmentation`: rotate, transform afín, contraste y brillo con Pillow. | CPU |
| 10 | Normalización y layout | `np.asarray(float32) / 255` y `transpose(2, 0, 1)` (HWC → CHW). | CPU |
| 11 | Ensamblado del lote | `default_collate`: apila 64 tensores en `[64, 3, 64, 64]`. | CPU |
| 12 | Forward | Convoluciones 3×3, BatchNorm, ReLU, MaxPool, capas lineales y logits `[64, 29]`. | CPU (hilos de PyTorch) |
| 13 | Pérdida | `CrossEntropyLoss` (log-softmax + NLL). | CPU |
| 14 | Backward | Gradientes de todas las capas (convoluciones transpuestas y productos de matrices). | CPU (hilos de PyTorch) |
| 15 | Actualización | `AdamW.step()`: momentos de 1.er/2.º orden y decaimiento de pesos para cada parámetro. | CPU |
| 16 | Métricas de entrenamiento | `loss.item()` y `argmax == labels` acumulados por lote. | CPU |
| 17 | Validación por época | Operaciones 7, 8, 10–12 sobre validation + accuracy y macro F1 (sklearn). | CPU |
| 18 | Selección del mejor estado | `copy.deepcopy(state_dict)` si mejora el macro F1; contador de early stopping. | CPU |
| 19 | Evaluación final | Se carga el mejor estado y se hace inferencia sobre test, más las métricas. | CPU |
| 20 | Persistencia | `torch.save` del checkpoint, JSON de métricas, CSV de predicciones y gráficas. | CPU + disco |

## 1.4 Operaciones repetitivas

Conteos para **un experimento** con la configuración del lab (12 180 / 2 610 / 2 610 imágenes, batch 64,
6 épocas): 191 lotes de train por época (190 completos + 1 de 20) y 41 lotes de validation y de test.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Lectura + decodificación + resize (ops. 2–6) | JPEG 200×200×3 → `uint8` 64×64×3 | **17 400** imágenes (1 vez; en el lab la caché se reutiliza entre experimentos) |
| `__getitem__`: caché → `/255` → CHW (ops. 8, 10) | 64×64×3 = 12 288 valores por imagen | 12 180 × 6 + 2 610 × 6 + 2 610 = **91 350** |
| Augmentation PIL (op. 9, experimentos `*_aug`) | Imagen 64×64 RGB, 4 transformaciones | 12 180 × 6 = **73 080** (292 320 operaciones de Pillow) |
| Ensamblado de lote (op. 11) | 64 tensores `[3, 64, 64]` | 191 × 6 + 41 × 7 = **1 433** lotes |
| Forward + pérdida + backward + AdamW (ops. 12–15) | Lote `[64, 3, 64, 64]` | 191 × 6 = **1 146** pasos de optimización |
| Convolución 3×3 (forward de `CNNRegularized`) | 6 capas conv, **22,7 M MAC por imagen** | 6 capas × 1 433 lotes; ≈ 22,7 M × (73 080 + 18 270) ≈ **2,1 × 10¹² MAC** solo en forward |
| Actualización AdamW por parámetro | 143 177 parámetros (`CNNRegularized`) / 815 805 (`CNNBaseline`) | 1 146 × 143 177 ≈ **1,6 × 10⁸** actualizaciones |
| Inferencia de validación (op. 17) | 2 610 imágenes | **6** pasadas (una por época) |
| Cálculo de métricas sklearn | Vectores de 2 610 etiquetas | 6 (validación) + 1 (test) = **7** |
| `deepcopy(state_dict)` (op. 18) | Todos los pesos del modelo | hasta **7** (inicial + una por cada mejora) |
| HOG (SVM, `hog_descriptor`) | Imagen 64×64 en gris: 64 celdas × 9 bins y 49 bloques | 17 400 (+ 12 180 con augmentation), con bucles Python anidados de 8×8 y 7×7 por imagen |

### Costo de cómputo de los modelos (calculado con hooks de PyTorch)

| Modelo | Parámetros | MAC por imagen (forward) | Capas conv |
|---|---:|---:|---:|
| `CNNRegularized` (`cnn_reg_*`) | 143 177 | 22,66 M | 6 |
| `CNNBaseline` (`cnn_base_*`) | 815 805 | 12,00 M | 3 |
| `SimpleMLP` (`mlp_*`) | 3 182 621 | 3,18 M | 0 |

Un paso de entrenamiento cuesta aproximadamente 3 veces el forward (forward + backward respecto de
activaciones y de pesos). Para `cnn_reg_drop035`, 6 épocas equivalen a
≈ 22,66 M × 3 × 73 080 ≈ **5 × 10¹² MAC (~10 TFLOP)**. Esa es la carga que se delega a la GPU.

## 1.5 Observaciones que guían la paralelización

1. **Dos tipos de trabajo muy distintos.** El manejo de archivos (ops. 2–6, 8–9) es irregular,
   con mucha ramificación y E/S, e **independiente entre imágenes**. La matemática de la red (ops. 12–15)
   es densa, regular y de alta intensidad aritmética (las mismas operaciones sobre tensores grandes).
2. **El lab procesa los datos en serie.** Con `num_workers=0`, el mismo proceso que entrena tiene que
   preparar cada lote. Mientras decodifica o aumenta imágenes, el cálculo de la red espera, y al revés.
3. **La normalización `/255` + CHW se repite 91 350 veces** en el CPU, aunque es aritmética trivial
   elemento a elemento. Es candidata natural a un kernel de GPU.
4. **El SVM-HOG** (`LinearSVC` de liblinear) es un solver de CPU sin soporte CUDA. Su extracción de
   características puede paralelizarse en CPU, pero queda **fuera del alcance de CUDA**
   (ver `decisiones_alcance.md`).
