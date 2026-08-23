"""Genera las figuras de curvas de aprendizaje del informe.

Lee los ``results.csv`` que Ultralytics deja en cada carpeta de ``runs/`` (o los
CSV ya consolidados en ``resultados/``) y produce dos figuras:

  1. ``curvas_train_val.png`` -- pérdida de entrenamiento vs. mAP@0.5 de
     validación, por época, para las dos ramas (transfer secuencial y scratch).
  2. ``curvas_perdidas.png``  -- box_loss y cls_loss de entrenamiento.

Uso:
    python src/graficar_curvas.py --resultados resultados/ --salida informe/figuras/
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COL_TRANSFER = "#1b7f3b"
COL_SCRATCH = "#c0392b"


def _cargar(ruta):
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    # Compatibilidad con el results.csv nativo de Ultralytics
    renombres = {
        "metrics/mAP50(B)": "val_mAP50",
        "metrics/mAP50-95(B)": "val_mAP5095",
        "metrics/precision(B)": "val_P",
        "metrics/recall(B)": "val_R",
        "train/box_loss": "box_loss",
        "train/cls_loss": "cls_loss",
        "train/dfl_loss": "dfl_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
    }
    return df.rename(columns={k: v for k, v in renombres.items() if k in df.columns})


def _tiene_val_loss(df):
    return {"val_box_loss", "val_cls_loss"}.issubset(df.columns)


def main():
    ap = argparse.ArgumentParser(description="Curvas de aprendizaje del proyecto.")
    ap.add_argument("--resultados", default="resultados",
                    help="Carpeta con curvas_transfer.csv y curvas_scratch.csv")
    ap.add_argument("--salida", default="informe/figuras",
                    help="Carpeta donde escribir los PNG")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    tr = _cargar(os.path.join(args.resultados, "curvas_transfer.csv"))
    sc = _cargar(os.path.join(args.resultados, "curvas_scratch.csv"))

    # --- Figura 1: entrenamiento vs. validación -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    ax[0].plot(tr["epoch"], tr["box_loss"] + tr["cls_loss"], color=COL_TRANSFER,
               lw=2, label="TL secuencial · train")
    ax[0].plot(sc["epoch"], sc["box_loss"] + sc["cls_loss"], color=COL_SCRATCH,
               lw=2, ls="--", label="Desde cero · train")

    # Si el results.csv nativo de Ultralytics está disponible, superponemos la
    # pérdida de validación: es lo que permite ver sobreajuste, no solo la de train.
    if _tiene_val_loss(tr):
        ax[0].plot(tr["epoch"], tr["val_box_loss"] + tr["val_cls_loss"],
                   color=COL_TRANSFER, lw=1.3, alpha=.55, label="TL secuencial · val")
    if _tiene_val_loss(sc):
        ax[0].plot(sc["epoch"], sc["val_box_loss"] + sc["val_cls_loss"],
                   color=COL_SCRATCH, lw=1.3, alpha=.55, ls="--", label="Desde cero · val")

    titulo = "Pérdida (box + cls)" if _tiene_val_loss(tr) else "Pérdida de entrenamiento (box + cls)"
    ax[0].set_title(titulo)
    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("Pérdida")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].plot(tr["epoch"], tr["val_mAP50"], color=COL_TRANSFER, lw=2,
               label="TL secuencial - val")
    ax[1].plot(sc["epoch"], sc["val_mAP50"], color=COL_SCRATCH, lw=2, ls="--",
               label="Desde cero - val")
    ax[1].set_title("mAP@0.5 en validación")
    ax[1].set_xlabel("Época")
    ax[1].set_ylabel("mAP@0.5")
    ax[1].set_ylim(0, 0.6)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    f1 = os.path.join(args.salida, "curvas_train_val.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)

    # --- Figura 2: pérdidas desagregadas ------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (col, titulo) in enumerate([("box_loss", "train/box_loss"),
                                       ("cls_loss", "train/cls_loss")]):
        ax[i].plot(tr["epoch"], tr[col], color=COL_TRANSFER, lw=2,
                   label="TL secuencial (KITTI → SCZ)")
        ax[i].plot(sc["epoch"], sc[col], color=COL_SCRATCH, lw=2, ls="--",
                   label="Desde cero (scratch)")
        ax[i].set_title(f"Evolución de {titulo}")
        ax[i].set_xlabel("Época")
        ax[i].set_ylabel("Pérdida")
        ax[i].legend(fontsize=8)
        ax[i].grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(args.salida, "curvas_perdidas.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)

    print(f"Figuras escritas en:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
