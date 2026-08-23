# Detección de Comercio Informal en la Vía Pública — Santa Cruz de la Sierra

**Transfer learning secuencial COCO → KITTI → Santa Cruz con YOLO11n**

Proyecto final del módulo *Procesamiento de Imágenes y Visión Computacional* — Universidad Católica Boliviana "San Pablo", Santa Cruz.

**Equipo:** Ana Patricia Aban Monzon · Juan Pablo Meriles &nbsp;·&nbsp; agosto de 2026
**Docente:** Erick Antero Maraz Zuñiga

---

## Resumen del problema

Dada una fotografía tomada a nivel de calle, el sistema localiza con cajas delimitadoras los puestos y estructuras instalados en la vía pública (`street_stall`), las personas que los atienden (`street_vendor`) y el contexto vial y peatonal que los rodea. El uso previsto es el **pre-etiquetado** de imágenes: convertir la anotación manual de un relevamiento urbano en una tarea de revisión y corrección de cajas propuestas. El sistema describe la ocupación observable del espacio público; no identifica personas ni infiere condición legal alguna de la actividad detectada.

## Resultados principales

Conjunto de prueba: 21 imágenes, 173 instancias. Ambas ramas con hiperparámetros y semilla idénticos.

| Métrica | Desde cero (control) | **TL secuencial** |
|---|---:|---:|
| mAP@0.5 | 0.1430 | **0.4164** |
| mAP@0.5:0.95 | 0.0483 | **0.2036** |
| Precisión (P) | 0.6224 | 0.4882 |
| Recall (R) | 0.1312 | **0.4227** |

Por clase, el modelo secuencial alcanza mAP@0.5 de **0.552** en `street_stall` y **0.394** en `street_vendor`. El informe completo, con el análisis de errores, está en [`informe/Informe_Deteccion_Comercio_Informal_SCZ.pdf`](informe/Informe_Deteccion_Comercio_Informal_SCZ.pdf) (fuente LaTeX en [`informe/main.tex`](informe/main.tex)).

> La precisión más alta del modelo desde cero es un artefacto de su recall de 0.13: emite muy pocas detecciones. Para pre-etiquetado, el recall es la métrica que decide la utilidad.

### Verificación de despliegue en CPU

`notebooks/inferencia_cpu.ipynb`, ejecutado de principio a fin en una laptop con `device="cpu"` (PyTorch 2.5.1, CUDA disponible pero deliberadamente ignorado):

| | |
|---|---|
| Checkpoint | `best_scz_transfer.pt` (5.47 MB), cargado con `map_location="cpu"` sin error |
| Parámetros | 2 591 595 (YOLO11n con cabeza de 9 clases) |
| Imágenes procesadas | 16, de `datos/muestra/` |
| **Tiempo medio de inferencia** | **313 ms por imagen** |
| Objetos anotados vs. detectados | 143 vs. 138 (umbral de confianza 0.25) |

Detecciones por clase: `pedestrian` 45 · `street_stall` 40 · `motorbike` 14 · `street_vendor` 14 · `car` 13 · `van` 5 · `person_sitting` 4 · `truck` 3.

> El cociente 138/143 es un **conteo agregado, no un emparejamiento caja a caja**: no mide acierto. Lo que indica es que el modelo no está sistemáticamente sobre- ni sub-detectando, y que propone un volumen de cajas del orden del que produciría un anotador humano — la condición práctica para que sirva de asistente de pre-etiquetado. Las métricas reales de acierto son las de la tabla anterior.

---

## Dataset (mini-datasheet)

### Conjunto propio — Santa Cruz de la Sierra

| | |
|---|---|
| **Origen** | 220 fotografías tomadas por el equipo con teléfono móvil, agosto de 2026, en vías con actividad comercial de Santa Cruz de la Sierra. |
| **Anotación** | Manual, con [LabelImg](https://github.com/HumanSignal/labelImg), exportada a formato YOLO (`clase cx cy w h` normalizado). |
| **Tamaño** | 220 imágenes · 1 667 cajas. Train 154 / Val 44 / Test 22 (70/20/10). |
| **Clases (9)** | `0 car` · `1 van` · `2 truck` · `3 pedestrian` · `4 person_sitting` · `5 cyclist` · `6 street_vendor` · `7 street_stall` · `8 motorbike` |
| **Partición** | Aleatoria por imagen con semilla fija (`random.Random(0)` sobre la lista ordenada). **No estratificada** — con 220 imágenes y 9 clases no es viable. |
| **Licencia** | Imágenes de autoría propia del equipo. Se distribuyen para uso académico. |

**Balance de entrenamiento** (1 147 cajas): `pedestrian` 36.8 % · `street_stall` 32.1 % · `street_vendor` 10.8 % · `car` 8.4 % · `person_sitting` 4.4 % · `motorbike` 3.6 % · `van` 2.0 % · `truck` 1.5 % · `cyclist` 0.5 %.

**Limitaciones conocidas** (el detalle está en la sección 3.3 del informe):

- **Riesgo de fuga por grupo.** Las fotos se tomaron el mismo día en pocos tramos de calle; como la partición es aleatoria por imagen y no por escena, un mismo puesto físico puede aparecer en entrenamiento y en prueba. **El 0.4164 reportado es una cota superior optimista** del desempeño en calles no vistas.
- **Desbalance severo.** `cyclist` tiene 6 cajas en train y 0 en test; `van`, 2 en test. Sus métricas no son interpretables.
- **Sesgo de captura.** Todas las imágenes son diurnas, con buena luz y clima seco, tomadas desde la vereda opuesta. Sin condiciones nocturnas, de lluvia ni contraluz.
- **Ambigüedad de anotación.** `street_vendor` vs. `pedestrian` es una decisión de criterio (quien atiende vs. quien compra) y no es perfectamente consistente entre anotadores.
- **Un archivo corrupto** en el split de test (imagen truncada); Ultralytics lo descarta, la evaluación corre sobre 21 imágenes.

### Conjunto intermedio — KITTI

[KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) en la distribución de Ultralytics: 7 481 imágenes de escena vial (≈5 985 train / 1 496 val, 8 128 instancias), 8 clases. Licencia **CC BY-NC-SA 3.0**, uso académico. **Se descarga automáticamente** (~372 MB) al ejecutar la etapa 1 de `src/train.py`; no hay que hacer nada manualmente.

---

## Cómo reproducir

### Qué corre dónde

El entrenamiento se hizo en Google Colab con GPU; el resto funciona en cualquier laptop sin GPU.

| Script | Dónde | Cuánto tarda |
|---|---|---|
| `src/train.py` | Colab / máquina con GPU | ≈44 min en Tesla T4. En CPU no es viable. |
| `src/evaluate.py` | CPU | < 1 min |
| `src/predict.py` | CPU | segundos por imagen |
| `src/analisis_errores.py` | CPU | < 2 min |
| `src/graficar_curvas.py` | CPU | instantáneo |
| `notebooks/inferencia_cpu.ipynb` | CPU, **sin GPU a propósito** | < 2 min |
| `notebooks/entrenamiento_completo.ipynb` | Colab | registro del experimento original; monta Google Drive y no corre fuera de Colab |

### 1. Instalación

```bash
git clone https://github.com/patyaban/Deteccion-Comercio-Informal-SCZ.git
cd Deteccion-Comercio-Informal-SCZ
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

En **Windows con PowerShell**, si `activate` falla con *"la ejecución de scripts está deshabilitada en este sistema"*, habilitarla solo para esa terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Para una máquina **sin GPU**, instalar la rueda de CPU de PyTorch antes del resto — pesa ~200 MB en vez de ~2.5 GB:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

En VS Code, después de crear el entorno: <kbd>Ctrl+Shift+P</kbd> → **Python: Select Interpreter** → elegir el que dice `.venv`.

**Entornos usados.** El resultado reportado se obtuvo en Google Colab con Python 3.12, PyTorch 2.11.0 + CUDA 12.8, Ultralytics 8.4.123 y GPU NVIDIA Tesla T4 — son las versiones fijadas en `requirements.txt`. La verificación en CPU de la sección anterior se corrió aparte, en una laptop con PyTorch 2.5.1: los pesos cargan y la inferencia funciona igual en ambas versiones.

### 2. Descargar los datos y los pesos

Los archivos grandes no están versionados (ver `.gitignore`). Están en Google Drive, accesibles con el enlace:

| Carpeta en Drive | Contenido | Dónde va |
|---|---|---|
| [**Modelos entrenados**](https://drive.google.com/drive/folders/1nMRF6Mn_mO8xNuqjh0hyJJB_vOD4YhMm) | `best_scz_final_step2.pt` (modelo final), `best_scratch.pt` (control), `best_kitti_step1.pt` (etapa intermedia), 5.4 MB c/u | `modelos/` |
| [**Dataset**](https://drive.google.com/drive/folders/1hKjRAwuSWIUk4w98mUjPP1caiwt-TeVQ?usp=sharing) | 220 fotografías `.jpg` con sus anotaciones `.txt` en formato YOLO | `datos/crudo/` |

Al copiar los pesos a `modelos/`, **renombrarlos** con los nombres que esperan los scripts:

| Nombre en Drive | Nombre en `modelos/` |
|---|---|
| `best_scz_final_step2.pt` | `best_scz_transfer.pt` |
| `best_scratch.pt` | `best_scz_scratch.pt` |
| `best_kitti_step1.pt` | `best_kitti_etapa1.pt` |

Por línea de comandos, con `gdown` (incluido en `requirements.txt`):

```bash
gdown --folder "https://drive.google.com/drive/folders/1nMRF6Mn_mO8xNuqjh0hyJJB_vOD4YhMm" -O modelos/
gdown --folder "https://drive.google.com/drive/folders/1hKjRAwuSWIUk4w98mUjPP1caiwt-TeVQ" -O datos/crudo/
```

`src/evaluate.py` y `src/predict.py` también aceptan un enlace de Drive directamente en `--modelo` y descargan los pesos por su cuenta.

En `datos/muestra/` hay 16 imágenes de ejemplo con sus etiquetas, versionadas en el repo y suficientes para correr el notebook de inferencia sin descargar nada.

### 3. Preparación de los datos

```bash
python src/preparar_datos.py --origen datos/crudo --salida datos/dataset_scz_yolo
```

Arma la partición 70/20/10 con semilla 0 y escribe `datos/dataset_scz_yolo/scz_data.yaml` con **rutas relativas** al repositorio. La plantilla del esquema de 9 clases está en [`configs/scz_data.yaml`](configs/scz_data.yaml). `src/train.py` ejecuta este paso por su cuenta si el dataset todavía no existe.

### 4. Entrenamiento completo

```bash
python src/train.py --datos datos/crudo --salida modelos/
```

Corrido **sin más argumentos**, esto reproduce exactamente el resultado del informe: prepara la partición 70/20/10 con semilla 0, entrena la etapa KITTI (10 épocas), la etapa Santa Cruz con `freeze=10` (80 épocas) y el control desde cero (80 épocas). Tiempo total en una Tesla T4: **≈44 minutos**.

Etapas por separado:

```bash
python src/train.py --datos datos/crudo --etapa kitti
python src/train.py --etapa scz --pesos-kitti modelos/best_kitti_etapa1.pt
python src/train.py --etapa scratch
```

### 5. Evaluación

```bash
python src/evaluate.py \
    --modelo modelos/best_scz_transfer.pt \
    --modelo-control modelos/best_scz_scratch.pt \
    --yaml datos/dataset_scz_yolo/scz_data.yaml \
    --split test --csv resultados/metricas_test.csv
```

Imprime las métricas por clase de cada modelo y la tabla comparativa del informe.

### 6. Inferencia sin GPU

**Notebook (obligatorio en la entrega):** [`notebooks/inferencia_cpu.ipynb`](notebooks/inferencia_cpu.ipynb) — carga el checkpoint con `torch.load(..., map_location="cpu")`, verifica explícitamente que se deserializa sin CUDA, corre inferencia sobre las imágenes de `datos/muestra/`, compara el número de detecciones con las anotaciones de referencia y mide el tiempo por imagen. Localiza la raíz del repositorio por su cuenta, así que funciona igual si se ejecuta desde `notebooks/` o desde la raíz. Ejecutar con **Restart & Run All**; resultados de la última corrida en la sección [Verificación de despliegue en CPU](#verificación-de-despliegue-en-cpu).

**Línea de comandos:**

```bash
python src/predict.py --modelo modelos/best_scz_transfer.pt \
                      --fuente datos/muestra/images --salida salidas/ --dispositivo cpu
python src/predict.py --modelo modelos/best_scz_transfer.pt --fuente video_scz.mp4 --dispositivo cpu
```

### 7. Figuras y análisis de errores

```bash
python src/graficar_curvas.py --resultados resultados/ --salida informe/figuras/
python src/analisis_errores.py --modelo modelos/best_scz_transfer.pt --split test \
                               --salida resultados/analisis
```

El primero regenera `informe/figuras/curvas_train_val.png` a partir de los CSV de `resultados/`. El segundo empareja predicciones y anotaciones por IoU y escribe en `resultados/analisis/`: la lámina `analisis_errores.png` con los ejemplos de aciertos y fallos, la planilla `analisis_errores.xlsx` con un caso por fila y su hipótesis, y los recortes en `analisis/aciertos/`.

### Reproducibilidad

- Semilla **0** fijada en `random`, `numpy`, `torch` y `torch.cuda` al inicio de `train.py` y `evaluate.py`; `deterministic=True` en las llamadas a Ultralytics.
- La partición se baraja sobre la lista **ordenada alfabéticamente**, no sobre el orden del sistema de archivos: dos ejecuciones de `preparar_datos.py` dan exactamente los mismos splits, en cualquier máquina.
- Los hiperparámetros del resultado reportado son los `default` de `argparse`.
- **Aviso sobre la corrida original:** el experimento en Colab barajó la lista tal como la devolvía `glob.glob()`, cuyo orden depende del sistema de archivos. Por eso la partición que genera `preparar_datos.py` **no es idéntica** a la que produjo las métricas reportadas, aunque use la misma semilla. Consecuencia práctica: los pesos publicados corresponden a la partición original, así que **no hay que evaluarlos contra una partición regenerada** — las imágenes se mezclarían entre entrenamiento y prueba. Para una cadena completamente coherente hay que volver a correr `train.py` sobre la partición nueva; a partir de ahí todo es reproducible de punta a punta.
- **Limitación:** con `amp=True` en GPU, algunas operaciones de cuDNN no son deterministas bit a bit. Dos corridas en la misma máquina dan resultados prácticamente idénticos, pero no garantizamos igualdad exacta al último decimal. En CPU la evaluación sí es determinista.

---

## Estructura del repositorio

```
Deteccion-Comercio-Informal-SCZ/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── scz_data.yaml                  # esquema de 9 clases (plantilla, rutas relativas)
├── datos/
│   └── muestra/                       # muestra chica versionada
│       ├── images/
│       └── labels/
├── informe/                           # Entregable A
│   ├── main.tex                       # fuente LaTeX (compila con pdfLaTeX)
│   ├── ucb-logo.png                   # logo de la portada
│   ├── Informe_Deteccion_Comercio_Informal_SCZ.pdf
│   └── figuras/
│       ├── curvas_train_val.png
│       ├── prediccion_1.png
│       └── prediccion_2.png
├── notebooks/
│   ├── entrenamiento_completo.ipynb   # notebook original del experimento (Colab)
│   └── inferencia_cpu.ipynb           # obligatorio — CPU only
├── resultados/
│   ├── curvas_transfer.csv            # métricas por época, rama transfer
│   ├── curvas_scratch.csv             # métricas por época, rama scratch
│   ├── curvas_kitti.csv               # métricas por época, etapa KITTI
│   └── analisis/
│       ├── analisis_errores.png       # lámina de aciertos y fallos
│       ├── analisis_errores.xlsx      # un caso por fila, con hipótesis
│       └── aciertos/                  # recortes de detecciones correctas
└── src/
    ├── preparar_datos.py              # partición 70/20/10 + scz_data.yaml
    ├── train.py                       # entrenamiento (CLI)
    ├── evaluate.py                    # métricas sobre test/val (CLI)
    ├── predict.py                     # inferencia imagen/carpeta/video (CLI)
    ├── graficar_curvas.py             # figuras del informe
    └── analisis_errores.py            # aciertos/fallos para la rúbrica
```

Las carpetas `modelos/`, `datos/crudo/`, `datos/dataset_scz_yolo/`, `runs/` y `salidas/` no están versionadas: se crean al descargar los pesos o al correr los scripts.

## Limitaciones conocidas del modelo

- El desempeño reportado se apoya en un conjunto de prueba de 21 imágenes; los intervalos de confianza son amplios y el riesgo de fuga por escena está presente (ver datasheet).
- `street_vendor` (mAP@0.5 = 0.394) es la clase más débil: un vendedor es visualmente un peatón y lo que lo distingue es su relación espacial con la mercadería, señal que un detector de una etapa con 124 ejemplos no llega a modelar.
- Las dos ramas quedaron en regímenes opuestos: la transferida muestra **sobreajuste incipiente** —su pérdida de validación deja de mejorar en la época 43, aunque el mAP sigue subiendo hasta el final—, mientras que la rama desde cero está **subentrenada**: su pérdida de validación todavía caía en la época 80.
- El modelo no ha visto escenas nocturnas, con lluvia ni a contraluz.

## Licencias

- **Imágenes del dataset propio:** autoría del equipo, distribuidas para uso académico.
- **KITTI:** CC BY-NC-SA 3.0 — uso no comercial.
- **Ultralytics YOLO11:** AGPL-3.0. Los pesos derivados heredan esa licencia.

## Referencias

Khanam & Hussain (2024), *YOLOv11: An Overview of the Key Architectural Enhancements*, arXiv:2410.17725 · Geiger, Lenz & Urtasun (2012), *The KITTI Vision Benchmark Suite*, CVPR · Lin et al. (2014), *Microsoft COCO*, ECCV · Yosinski et al. (2014), *How transferable are features in deep neural networks?*, NeurIPS · Gebru et al. (2018), *Datasheets for Datasets*, arXiv:1803.09010.
