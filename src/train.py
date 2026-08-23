"""Entrenamiento del detector de comercio informal (YOLO11n).

Reproduce el experimento completo del informe:

    Etapa 1 (kitti)    COCO -> KITTI      yolo11n.pt,  10 épocas, backbone libre
    Etapa 2 (scz)      KITTI -> SantaCruz best.pt,     80 épocas, freeze=10
    Control (scratch)  yolo11n.yaml       desde cero,  80 épocas, sin pesos previos

Ejecutar sin argumentos (más allá de --datos) reproduce exactamente el resultado
reportado en el informe: semilla 0, imgsz 640, batch 16, SGD.

Uso típico:
    python src/train.py --datos datos/crudo --salida modelos/
    python src/train.py --datos datos/crudo --salida modelos/ --etapa scratch
"""

import argparse
import os
import random
import shutil

import numpy as np
import torch
from ultralytics import YOLO

from preparar_datos import preparar


def fijar_semilla(semilla=0):
    """Fija todas las fuentes de aleatoriedad del pipeline."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def entrenar_kitti(args, project):
    """Etapa 1: adaptación de dominio vial. COCO -> KITTI.

    Ultralytics descarga kitti.zip automáticamente la primera vez (~372 MB)."""
    print("[Etapa 1] Adaptación de dominio vial (yolo11n.pt -> kitti.yaml)")
    modelo = YOLO("yolo11n.pt")
    modelo.train(
        data="kitti.yaml",
        epochs=args.epocas_kitti,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizador,
        seed=args.semilla,
        deterministic=True,
        project=project,
        name="kitti_pretrain",
        exist_ok=True,
        verbose=True,
    )
    return os.path.join(project, "kitti_pretrain", "weights", "best.pt")


def entrenar_scz(args, project, pesos_iniciales):
    """Etapa 2: especialización en comercio informal, con backbone congelado."""
    print(f"[Etapa 2] Especialización en Santa Cruz desde {pesos_iniciales} (freeze={args.freeze})")
    modelo = YOLO(pesos_iniciales)
    modelo.train(
        data=args.yaml,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizador,
        freeze=args.freeze,
        seed=args.semilla,
        deterministic=True,
        project=project,
        name="tl_secuencial_scz",
        exist_ok=True,
        verbose=True,
    )
    return os.path.join(project, "tl_secuencial_scz", "weights", "best.pt")


def entrenar_scratch(args, project):
    """Control: misma arquitectura y mismos hiperparámetros, sin pesos previos."""
    print("[Control] Entrenamiento desde cero (yolo11n.yaml)")
    modelo = YOLO("yolo11n.yaml")
    modelo.train(
        data=args.yaml,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizador,
        seed=args.semilla,
        deterministic=True,
        project=project,
        name="scratch",
        exist_ok=True,
        verbose=True,
    )
    return os.path.join(project, "scratch", "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datos", default=None,
                    help="Carpeta plana con imágenes y .txt. Si se indica, se "
                         "prepara el dataset YOLO antes de entrenar.")
    ap.add_argument("--yaml", default="datos/dataset_scz_yolo/scz_data.yaml",
                    help="Ruta al scz_data.yaml (se genera solo si se pasa --datos)")
    ap.add_argument("--dataset-yolo", default="datos/dataset_scz_yolo",
                    help="Carpeta destino del dataset en formato YOLO")
    ap.add_argument("--salida", default="modelos",
                    help="Carpeta donde se copian los pesos finales (.pt)")
    ap.add_argument("--runs", default="runs/detect/runs_scz",
                    help="Carpeta de trabajo de Ultralytics")
    ap.add_argument("--etapa", default="todo",
                    choices=["todo", "kitti", "scz", "scratch"],
                    help="Qué entrenar. 'todo' = kitti + scz + scratch")
    ap.add_argument("--pesos-kitti", default=None,
                    help="Pesos de la etapa KITTI ya entrenados (salta la etapa 1)")
    ap.add_argument("--epocas", type=int, default=80, help="Épocas de la etapa Santa Cruz")
    ap.add_argument("--epocas-kitti", type=int, default=10, help="Épocas de la etapa KITTI")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=10,
                    help="Nº de capas del backbone a congelar en la etapa 2")
    ap.add_argument("--optimizador", default="SGD")
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    fijar_semilla(args.semilla)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")
    if dispositivo == "cpu":
        print("AVISO: sin GPU el entrenamiento es muy lento. Para solo probar el "
              "modelo ya entrenado use src/predict.py o notebooks/inferencia_cpu.ipynb.")

    if args.datos:
        args.yaml = preparar(args.datos, args.dataset_yolo, args.semilla)

    if not os.path.exists(args.yaml) and args.etapa != "kitti":
        raise SystemExit(f"No existe '{args.yaml}'. Pase --datos para generarlo.")

    os.makedirs(args.salida, exist_ok=True)
    project = args.runs
    finales = {}

    pesos_kitti = args.pesos_kitti
    if args.etapa in ("todo", "kitti") and pesos_kitti is None:
        pesos_kitti = entrenar_kitti(args, project)
        finales["best_kitti_etapa1.pt"] = pesos_kitti

    if args.etapa in ("todo", "scz"):
        if pesos_kitti is None or not os.path.exists(pesos_kitti):
            raise SystemExit("La etapa 'scz' necesita --pesos-kitti o correr antes la etapa 'kitti'.")
        finales["best_scz_transfer.pt"] = entrenar_scz(args, project, pesos_kitti)

    if args.etapa in ("todo", "scratch"):
        finales["best_scz_scratch.pt"] = entrenar_scratch(args, project)

    print("\nPesos finales:")
    for nombre, ruta in finales.items():
        if os.path.exists(ruta):
            destino = os.path.join(args.salida, nombre)
            shutil.copy(ruta, destino)
            print(f"  {destino}")
        else:
            print(f"  (no encontrado) {ruta}")


if __name__ == "__main__":
    main()
