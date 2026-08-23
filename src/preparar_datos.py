"""Prepara el dataset de Santa Cruz en el formato de carpetas que espera YOLO.

Toma una carpeta plana donde cada imagen ``foto.jpg`` tiene al lado su
anotación ``foto.txt`` (formato YOLO: ``clase cx cy w h`` normalizado) y genera:

    <salida>/
        images/{train,val,test}/
        labels/{train,val,test}/
        scz_data.yaml

La partición es 70/20/10 con semilla fija, de modo que dos ejecuciones con la
misma semilla producen exactamente los mismos splits.

Uso:
    python src/preparar_datos.py --origen datos/crudo --salida datos/dataset_scz_yolo --semilla 0
"""

import argparse
import glob
import os
import random
import shutil
from collections import Counter

# Esquema de 9 clases: las 6 primeras heredan la semántica vial de KITTI,
# las 3 últimas son las clases propias del dominio de comercio informal.
CLASS_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "street_vendor",
    7: "street_stall",
    8: "motorbike",
}

EXTENSIONES = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def convertir_caja(xmin, ymin, xmax, ymax, w_img, h_img):
    """Convierte [xmin, ymin, xmax, ymax] en píxeles a YOLO [cx, cy, w, h] normalizado."""
    cx = ((xmin + xmax) / 2.0) / w_img
    cy = ((ymin + ymax) / 2.0) / h_img
    w = (xmax - xmin) / float(w_img)
    h = (ymax - ymin) / float(h_img)
    return cx, cy, w, h


def emparejar(origen):
    """Devuelve la lista de pares (imagen, etiqueta) que existen en disco."""
    imagenes = []
    for ext in EXTENSIONES:
        imagenes.extend(glob.glob(os.path.join(origen, ext)))
    pares = []
    huerfanas = []
    for img in sorted(imagenes):
        txt = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(txt):
            pares.append((img, txt))
        else:
            huerfanas.append(img)
    if huerfanas:
        print(f"AVISO: {len(huerfanas)} imágenes sin archivo .txt fueron descartadas.")
    return pares


def particionar(pares, semilla=0, p_train=0.70, p_val=0.20):
    """Partición aleatoria reproducible. Ordena antes de barajar para no depender
    del orden que devuelva el sistema de archivos."""
    pares = sorted(pares)
    rng = random.Random(semilla)
    rng.shuffle(pares)
    n = len(pares)
    n_train = int(n * p_train)
    n_val = int(n * p_val)
    return {
        "train": pares[:n_train],
        "val": pares[n_train:n_train + n_val],
        "test": pares[n_train + n_val:],
    }


def escribir_yaml(destino):
    ruta = os.path.join(destino, "scz_data.yaml")
    lineas = [
        f"path: {os.path.abspath(destino)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lineas += [f"  {k}: {v}" for k, v in CLASS_NAMES.items()]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    return ruta


def contar_clases(pares):
    conteo = Counter()
    for _, txt in pares:
        with open(txt, "r", encoding="utf-8") as f:
            for linea in f:
                partes = linea.strip().split()
                if partes:
                    conteo[int(partes[0])] += 1
    return conteo


def preparar(origen, salida, semilla=0):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(salida, "images", split), exist_ok=True)
        os.makedirs(os.path.join(salida, "labels", split), exist_ok=True)

    pares = emparejar(origen)
    if not pares:
        raise SystemExit(
            f"No se encontraron pares imagen/.txt en '{origen}'. "
            "Verifique la ruta y que cada imagen tenga su anotación al lado."
        )

    splits = particionar(pares, semilla=semilla)
    for split, items in splits.items():
        for img, txt in items:
            shutil.copy(img, os.path.join(salida, "images", split, os.path.basename(img)))
            shutil.copy(txt, os.path.join(salida, "labels", split, os.path.basename(txt)))
        print(f"Split '{split}': {len(items)} imágenes.")

    yaml_path = escribir_yaml(salida)
    print(f"YAML escrito en: {yaml_path}")

    conteo = contar_clases(splits["train"])
    total = sum(conteo.values())
    print("\nDistribución de cajas en TRAIN:")
    print(f"{'ID':<4}{'Clase':<18}{'Cajas':>8}{'%':>9}")
    for cid, nombre in CLASS_NAMES.items():
        c = conteo.get(cid, 0)
        pct = (c / total * 100) if total else 0.0
        print(f"{cid:<4}{nombre:<18}{c:>8}{pct:>8.2f}%")
    print(f"{'':<22}{total:>8}   instancias en total")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--origen", required=True,
                    help="Carpeta plana con imágenes y sus .txt al lado")
    ap.add_argument("--salida", default="datos/dataset_scz_yolo",
                    help="Carpeta destino con la estructura YOLO")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    # Sanity check de la fórmula de conversión (debe dar 0.5, 0.5, 1.0, 1.0)
    assert convertir_caja(0, 0, 640, 480, 640, 480) == (0.5, 0.5, 1.0, 1.0)

    preparar(args.origen, args.salida, args.semilla)


if __name__ == "__main__":
    main()
