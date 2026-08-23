"""Evalúa un modelo guardado sobre el split de test (o val) y reporta métricas.

Si se pasan los dos modelos (transfer y scratch) imprime además la tabla
comparativa que aparece en el informe.

Uso:
    python src/evaluate.py --modelo modelos/best_scz_transfer.pt --yaml datos/dataset_scz_yolo/scz_data.yaml
    python src/evaluate.py --modelo modelos/best_scz_transfer.pt \
                           --modelo-control modelos/best_scz_scratch.pt --csv resultados/metricas_test.csv
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from ultralytics import YOLO

CLASES = ["car", "van", "truck", "pedestrian", "person_sitting",
          "cyclist", "street_vendor", "street_stall", "motorbike"]


def fijar_semilla(semilla=0):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def descargar_si_es_url(ruta, destino="modelos"):
    """Permite pasar una URL directa en --modelo; la descarga a ``destino``."""
    if not str(ruta).startswith(("http://", "https://")):
        return ruta
    import urllib.request
    os.makedirs(destino, exist_ok=True)
    local = os.path.join(destino, os.path.basename(ruta.split("?")[0]) or "modelo.pt")
    if not os.path.exists(local):
        print(f"Descargando pesos desde {ruta} ...")
        urllib.request.urlretrieve(ruta, local)
    return local


def evaluar(ruta_modelo, yaml_datos, split, dispositivo, nombre_run):
    modelo = YOLO(ruta_modelo)
    m = modelo.val(data=yaml_datos, split=split, device=dispositivo,
                   project="runs/detect/eval", name=nombre_run, exist_ok=True)
    # Solo se reportan las clases realmente presentes en el split: las ausentes
    # aparecerían como 0.000 y se leerían, erróneamente, como un fallo del modelo.
    filas = []
    presentes = list(getattr(m.box, "ap_class_index", range(len(CLASES))))
    for pos, cid in enumerate(presentes):
        cid = int(cid)
        nombre = CLASES[cid] if cid < len(CLASES) else str(cid)
        try:
            p, r, ap50, ap = m.box.class_result(pos)
        except (IndexError, KeyError):
            continue
        filas.append((nombre, p, r, ap50, ap))
    ausentes = [c for i, c in enumerate(CLASES) if i not in [int(x) for x in presentes]]
    if ausentes:
        print(f"(clases sin instancias en el split '{split}': {', '.join(ausentes)})")
    return {
        "global": (m.box.mp, m.box.mr, m.box.map50, m.box.map),
        "por_clase": filas,
    }


def imprimir(titulo, res):
    print(f"\n=== {titulo} ===")
    p, r, m50, m = res["global"]
    print(f"{'Clase':<18}{'P':>8}{'R':>8}{'mAP@.5':>10}{'mAP@.5:.95':>12}")
    print(f"{'TODAS':<18}{p:>8.3f}{r:>8.3f}{m50:>10.3f}{m:>12.3f}")
    for nombre, cp, cr, c50, c in res["por_clase"]:
        print(f"{nombre:<18}{cp:>8.3f}{cr:>8.3f}{c50:>10.3f}{c:>12.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True,
                    help="Ruta al .pt entrenado, o URL de descarga directa")
    ap.add_argument("--modelo-control", default=None,
                    help="Segundo modelo (scratch) para la tabla comparativa")
    ap.add_argument("--yaml", default="datos/dataset_scz_yolo/scz_data.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--csv", default=None, help="Ruta donde guardar las métricas")
    ap.add_argument("--dispositivo", default=None,
                    help="'cpu', '0', etc. Por defecto usa GPU si está disponible")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    fijar_semilla(args.semilla)
    dispositivo = args.dispositivo or ("0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.yaml):
        raise SystemExit(f"No existe '{args.yaml}'. Corra antes src/preparar_datos.py.")

    ruta = descargar_si_es_url(args.modelo)
    res_tl = evaluar(ruta, args.yaml, args.split, dispositivo, "transfer")
    imprimir(f"Transfer learning secuencial - split {args.split}", res_tl)

    res_ct = None
    if args.modelo_control:
        ruta_ct = descargar_si_es_url(args.modelo_control)
        res_ct = evaluar(ruta_ct, args.yaml, args.split, dispositivo, "scratch")
        imprimir(f"Desde cero (control) - split {args.split}", res_ct)

        print("\n=== Tabla comparativa (split %s) ===" % args.split)
        print(f"{'Métrica':<16}{'Desde cero':>14}{'TL secuencial':>16}")
        etiquetas = ["Precisión (P)", "Recall (R)", "mAP@0.5", "mAP@0.5:0.95"]
        orden = [0, 1, 2, 3]
        for et, i in zip(etiquetas, orden):
            print(f"{et:<16}{res_ct['global'][i]:>14.4f}{res_tl['global'][i]:>16.4f}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["modelo", "clase", "P", "R", "mAP50", "mAP50_95"])
            for etiqueta, res in [("transfer", res_tl), ("scratch", res_ct)]:
                if res is None:
                    continue
                p, r, m50, m = res["global"]
                w.writerow([etiqueta, "all", f"{p:.4f}", f"{r:.4f}", f"{m50:.4f}", f"{m:.4f}"])
                for nombre, cp, cr, c50, c in res["por_clase"]:
                    w.writerow([etiqueta, nombre, f"{cp:.4f}", f"{cr:.4f}", f"{c50:.4f}", f"{c:.4f}"])
        print(f"\nMétricas guardadas en {args.csv}")


if __name__ == "__main__":
    main()
