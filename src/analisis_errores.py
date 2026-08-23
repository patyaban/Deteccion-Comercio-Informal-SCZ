"""Análisis de errores: arma la lámina de aciertos y fallos que pide la rúbrica.

Recorre un conjunto de imágenes anotadas, empareja cada predicción con la anotación
de referencia por IoU y clasifica el resultado en cuatro categorías:

    acierto (VP)        IoU >= umbral y la clase coincide
    clase equivocada    IoU >= umbral pero la clase no coincide
    falso positivo      el modelo detectó algo que nadie anotó
    falso negativo      había una anotación y el modelo no detectó nada

Genera:

    <salida>/analisis_errores.png   lámina con 12 aciertos y 12 fallos (para el informe)
    <salida>/analisis_errores.csv   una fila por caso, con columna 'hipotesis' vacía
    <salida>/aciertos/  <salida>/fallos/    los recortes individuales

Acepta dos disposiciones de carpetas:

    <dataset>/images/<split>/  +  <dataset>/labels/<split>/     (dataset particionado)
    <dataset>/images/          +  <dataset>/labels/             (p. ej. datos/muestra)

Uso:
    python src/analisis_errores.py --modelo modelos/best_scz_transfer.pt \
                                   --dataset datos/muestra --salida resultados/analisis
"""

import argparse
import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CLASES = ["car", "van", "truck", "pedestrian", "person_sitting",
          "cyclist", "street_vendor", "street_stall", "motorbike"]

EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def iou(a, b):
    """IoU entre dos cajas [x1, y1, x2, y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def listar_imagenes(dir_img):
    if not os.path.isdir(dir_img):
        return []
    return sorted(os.path.join(dir_img, f) for f in os.listdir(dir_img)
                  if f.endswith(EXTS))


def resolver_carpetas(dataset, split):
    """Devuelve (dir_imagenes, dir_labels) soportando ambas disposiciones.

    Se queda con la primera carpeta que **contenga imágenes**, no con la primera
    que exista: un 'images/test/' vacío no debe ganarle a un 'images/' con fotos.
    """
    candidatas = []
    if split:
        candidatas.append((os.path.join(dataset, "images", split),
                           os.path.join(dataset, "labels", split)))
    candidatas.append((os.path.join(dataset, "images"),
                       os.path.join(dataset, "labels")))

    for dir_img, dir_lbl in candidatas:
        if listar_imagenes(dir_img):
            return dir_img, dir_lbl

    revisadas = "\n".join(f"    {d}  ({'existe pero está vacía' if os.path.isdir(d) else 'no existe'})"
                          for d, _ in candidatas)
    raise SystemExit(f"No encontré imágenes en '{dataset}'. Revisé:\n{revisadas}")


def leer_etiquetas(ruta_txt, w, h):
    cajas = []
    if not os.path.exists(ruta_txt):
        return cajas
    with open(ruta_txt, "r", encoding="utf-8") as f:
        for linea in f:
            p = linea.strip().split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            cajas.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return cajas


def recortar(img, caja, margen=0.18):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = caja
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def _cuadrar(crop, lado=320, fondo=(238, 238, 238)):
    """Encaja el recorte en un lienzo cuadrado sin deformarlo.

    Los recortes tienen proporciones muy distintas (un peatón es alto y angosto,
    un puesto es ancho y bajo). Sin esto la grilla queda despareja e ilegible.
    """
    import numpy as np
    h, w = crop.shape[:2]
    esc = min(lado / w, lado / h)
    nw, nh = max(1, int(w * esc)), max(1, int(h * esc))
    interp = cv2.INTER_AREA if esc < 1 else cv2.INTER_CUBIC
    redim = cv2.resize(crop, (nw, nh), interpolation=interp)
    lienzo = np.full((lado, lado, 3), fondo, dtype=crop.dtype)
    y0, x0 = (lado - nh) // 2, (lado - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = redim
    return lienzo


def _bloque(subfig, items, titulo, color, por_fila, filas):
    subfig.suptitle(titulo, fontsize=12, weight="bold", color=color, y=0.985)
    ejes = subfig.subplots(filas, por_fila)
    ejes = ejes.ravel() if hasattr(ejes, "ravel") else [ejes]
    for k, ax in enumerate(ejes):
        ax.axis("off")
        if k < len(items):
            crop, etq = items[k]
            ax.imshow(cv2.cvtColor(_cuadrar(crop), cv2.COLOR_BGR2RGB))
            ax.set_title(etq, fontsize=7.5, color=color, pad=3)


def lamina(aciertos, fallos, ruta_png, por_fila=6, filas=2):
    """Arma la figura de dos bloques que va al informe."""
    fig = plt.figure(figsize=(13, 9.6))
    arriba, abajo = fig.subfigures(2, 1, hspace=0.04)
    _bloque(arriba, aciertos, f"ACIERTOS — {len(aciertos)} verdaderos positivos",
            "#16702F", por_fila, filas)
    _bloque(abajo, fallos, f"FALLOS — {len(fallos)} casos",
            "#B32332", por_fila, filas)
    fig.savefig(ruta_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--dataset", default="datos/muestra")
    ap.add_argument("--split", default="test",
                    help="Subcarpeta de images/. Se ignora si el dataset es plano.")
    ap.add_argument("--salida", default="resultados/analisis")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-umbral", type=float, default=0.5)
    ap.add_argument("--max-ejemplos", type=int, default=12,
                    help="Cuántos aciertos y cuántos fallos exportar a la lámina")
    ap.add_argument("--dispositivo", default="cpu")
    args = ap.parse_args()

    dir_img, dir_lbl = resolver_carpetas(args.dataset, args.split)
    dir_ac = os.path.join(args.salida, "aciertos")
    dir_fa = os.path.join(args.salida, "fallos")
    os.makedirs(dir_ac, exist_ok=True)
    os.makedirs(dir_fa, exist_ok=True)

    imagenes = listar_imagenes(dir_img)
    if not imagenes:
        raise SystemExit(f"Sin imágenes en {dir_img}")
    print(f"Imágenes: {len(imagenes)}   Anotaciones: {dir_lbl}")

    modelo = YOLO(args.modelo)
    aciertos, fallos, filas = [], [], []

    for ruta in imagenes:
        img = cv2.imread(ruta)
        if img is None:
            print(f"AVISO: no se pudo leer {os.path.basename(ruta)}")
            continue
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(ruta))[0]
        gt = leer_etiquetas(os.path.join(dir_lbl, base + ".txt"), w, h)

        r = modelo.predict(source=ruta, conf=args.conf,
                           device=args.dispositivo, verbose=False)[0]
        usados = set()

        for b in r.boxes:
            cid_p = int(b.cls)
            conf = float(b.conf)
            caja = [float(v) for v in b.xyxy[0]]

            mejor, mejor_iou = -1, 0.0
            for j, (_, caja_g) in enumerate(gt):
                v = iou(caja, caja_g)
                if v > mejor_iou:
                    mejor, mejor_iou = j, v

            emparejado = mejor_iou >= args.iou_umbral and mejor >= 0
            crop = recortar(img, caja)

            if emparejado and gt[mejor][0] == cid_p and mejor not in usados:
                usados.add(mejor)
                tipo = "acierto (VP)"
                real = CLASES[cid_p]
                if crop is not None and len(aciertos) < args.max_ejemplos:
                    aciertos.append((crop, f"{CLASES[cid_p]}  {conf:.2f}"))
                carpeta = dir_ac
            else:
                if emparejado and gt[mejor][0] == cid_p:
                    # La clase es correcta, pero otra predicción de mayor confianza ya
                    # reclamó esa anotación: son dos cajas sobre el mismo objeto que
                    # el NMS no llegó a fusionar.
                    real = CLASES[cid_p]
                    tipo = "deteccion duplicada"
                    etiqueta = f"{CLASES[cid_p]} {conf:.2f}\ncaja duplicada"
                elif emparejado:
                    real = CLASES[gt[mejor][0]]
                    tipo = f"clase equivocada (era {real})"
                    etiqueta = f"predijo {CLASES[cid_p]} {conf:.2f}\nera {real}"
                else:
                    real = "-"
                    tipo = "falso positivo"
                    etiqueta = f"{CLASES[cid_p]} {conf:.2f}\nfalso positivo"
                if crop is not None and len(fallos) < args.max_ejemplos:
                    fallos.append((crop, etiqueta))
                carpeta = dir_fa

            if crop is not None:
                nombre = f"{base}_{CLASES[cid_p]}_{conf:.2f}_{len(filas):03d}.jpg"
                cv2.imwrite(os.path.join(carpeta, nombre), crop)
            else:
                nombre = "(recorte demasiado pequeño)"

            filas.append({
                "archivo": nombre, "imagen": os.path.basename(ruta), "tipo": tipo,
                "clase_predicha": CLASES[cid_p], "clase_real": real,
                "confianza": f"{conf:.3f}", "iou": f"{mejor_iou:.3f}", "hipotesis": "",
            })

        # Falsos negativos: anotaciones que ninguna predicción cubrió
        for j, (cid_g, caja_g) in enumerate(gt):
            if j in usados:
                continue
            crop = recortar(img, caja_g)
            if crop is not None and len(fallos) < args.max_ejemplos:
                fallos.append((crop, f"{CLASES[cid_g]}\nno detectado"))
            nombre = f"{base}_FN_{CLASES[cid_g]}_{len(filas):03d}.jpg"
            if crop is not None:
                cv2.imwrite(os.path.join(dir_fa, nombre), crop)
            filas.append({
                "archivo": nombre, "imagen": os.path.basename(ruta),
                "tipo": "falso negativo", "clase_predicha": "-",
                "clase_real": CLASES[cid_g], "confianza": "-",
                "iou": "0.000", "hipotesis": "",
            })

    ruta_png = os.path.join(args.salida, "analisis_errores.png")
    lamina(aciertos, fallos, ruta_png)

    ruta_csv = os.path.join(args.salida, "analisis_errores.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=["archivo", "imagen", "tipo", "clase_predicha",
                                           "clase_real", "confianza", "iou", "hipotesis"])
        w_.writeheader()
        w_.writerows(filas)

    resumen = {}
    for fila in filas:
        clave = fila["tipo"].split(" (")[0]
        resumen[clave] = resumen.get(clave, 0) + 1

    print("\nResumen de casos analizados")
    for k, v in sorted(resumen.items(), key=lambda x: -x[1]):
        print(f"  {k:<24}{v:>5}")
    print(f"\nEn la lámina: {len(aciertos)} aciertos y {len(fallos)} fallos")
    print(f"  {ruta_png}   <- va al informe (informe/figuras/)")
    print(f"  {ruta_csv}   <- completar la columna 'hipotesis'")


if __name__ == "__main__":
    main()