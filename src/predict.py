"""Inferencia con el modelo entrenado sobre una imagen, una carpeta o un video.

Funciona en CPU sin GPU (usa `map_location="cpu"` implícitamente vía device="cpu").

Uso:
    python src/predict.py --modelo modelos/best_scz_transfer.pt --fuente datos/muestra/images --salida salidas/
    python src/predict.py --modelo URL_DEL_MODELO --fuente foto.jpg --dispositivo cpu
    python src/predict.py --modelo modelos/best_scz_transfer.pt --fuente video.mp4 --salida salidas/
"""

import argparse
import glob
import os

import torch
from ultralytics import YOLO

EXTS_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EXTS_VID = (".mp4", ".avi", ".mov", ".mkv")


def descargar_si_es_url(ruta, destino="modelos"):
    if not str(ruta).startswith(("http://", "https://")):
        return ruta
    import urllib.request
    os.makedirs(destino, exist_ok=True)
    local = os.path.join(destino, os.path.basename(ruta.split("?")[0]) or "modelo.pt")
    if not os.path.exists(local):
        print(f"Descargando pesos desde {ruta} ...")
        urllib.request.urlretrieve(ruta, local)
    return local


def listar_fuentes(fuente):
    if os.path.isdir(fuente):
        archivos = []
        for ext in EXTS_IMG:
            archivos += glob.glob(os.path.join(fuente, f"*{ext}"))
            archivos += glob.glob(os.path.join(fuente, f"*{ext.upper()}"))
        return sorted(archivos)
    return [fuente]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True, help="Ruta al .pt o URL de descarga directa")
    ap.add_argument("--fuente", required=True, help="Imagen, carpeta de imágenes o video")
    ap.add_argument("--salida", default="salidas", help="Carpeta donde escribir las imágenes anotadas")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    ap.add_argument("--iou", type=float, default=0.7, help="Umbral de IoU para NMS")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dispositivo", default=None, help="'cpu' o '0'. Por defecto CPU si no hay GPU")
    ap.add_argument("--sin-guardar", action="store_true",
                    help="Solo imprime las detecciones, no escribe imágenes")
    args = ap.parse_args()

    dispositivo = args.dispositivo or ("0" if torch.cuda.is_available() else "cpu")
    ruta = descargar_si_es_url(args.modelo)
    modelo = YOLO(ruta)
    os.makedirs(args.salida, exist_ok=True)

    es_video = str(args.fuente).lower().endswith(EXTS_VID)
    fuentes = [args.fuente] if es_video else listar_fuentes(args.fuente)
    if not fuentes:
        raise SystemExit(f"No se encontraron archivos en '{args.fuente}'.")

    print(f"Modelo: {ruta}\nDispositivo: {dispositivo}\nArchivos: {len(fuentes)}\n")

    total = 0
    for f in fuentes:
        resultados = modelo.predict(
            source=f, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=dispositivo, save=not args.sin_guardar,
            project=args.salida, name="predicciones", exist_ok=True,
            stream=es_video, verbose=False,
        )
        if es_video:
            n = sum(len(r.boxes) for r in resultados)
            print(f"{os.path.basename(f)}: {n} detecciones en total")
            total += n
            continue
        for r in resultados:
            nombres = r.names
            detecciones = [
                f"{nombres[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes
            ]
            total += len(detecciones)
            print(f"{os.path.basename(f)}: {len(detecciones)} detecciones"
                  + (f" -> {', '.join(detecciones)}" if detecciones else ""))

    print(f"\nTotal de detecciones: {total}")
    if not args.sin_guardar:
        print(f"Imágenes anotadas en: {os.path.join(args.salida, 'predicciones')}")


if __name__ == "__main__":
    main()
