#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para comparar curvas ROC medias ± desviación estándar entre un modelo Deep Learning y Radiomics.

Este script:
- Carga predicciones de Deep Learning desde un CSV con columnas 'split', 'true_label', 'prob_class_1'.
- Carga predicciones de Radiomics desde un CSV con columnas 'Fold', 'Classifier', 'y_val', 'y_prob'.
- Calcula ROC medias y desviaciones por fold para cada enfoque.
- Dibuja ambas curvas en un único gráfico, con sombreado ± desviación.
"""

from __future__ import annotations
import argparse, ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots                         
from sklearn import metrics

plt.style.use(["science", "grid"])
DPI = 300
COLOR_DL, COLOR_RAD = "#0072B2", "#D55E00"


# --------------------------- utilidades --------------------------- #

def _parse(series: pd.Series):
    """
    Parsea una serie con listas guardadas como strings a objetos Python.

    Args:
        series (pd.Series): Serie de pandas cuyos elementos pueden ser strings de listas.
    Returns:
        pd.Series: Serie con listas evaluadas en lugar de strings.
    """
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def load_radiomics(csv_path: Path, classifier: str):
    """
    Carga predicciones de Radiomics de un CSV.

    Filtra filas donde 'Classifier' coincide y 'Repeat' == 1.
    Convierte columnas 'y_val' y 'y_prob' de strings a listas.
    Construye un dict fold -> (y_true, y_prob).

    Args:
        csv_path (Path): Ruta al CSV con columnas 'Fold','Classifier','Repeat','y_val','y_prob'.
        classifier (str): Nombre del clasificador a filtrar en la columna 'Classifier'.
    Returns:
        dict[int, tuple[np.ndarray, np.ndarray]]: Mapea cada Fold a arrays numpy de y_true y y_prob.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["Classifier"] == classifier) & (df["Repeat"] == 1)]
    df["y_val"], df["y_prob"] = _parse(df["y_val"]), _parse(df["y_prob"])
    return {int(r["Fold"]): (r["y_val"], r["y_prob"]) for _, r in df.iterrows()}

def load_dl(csv_path: Path):
    """
    Carga predicciones de Deep Learning de un CSV.

    El CSV debe tener columnas 'split', 'true_label', 'prob_class_1'.
    Agrupa por split y retorna dict split -> (y_true, y_prob).

    Args:
        csv_path (Path): Ruta al CSV.
    Returns:
        dict[int, tuple[np.ndarray, np.ndarray]]: Mapea cada split a arrays numpy.
    """
    df = pd.read_csv(csv_path)
    return {int(fid): (g["true_label"].values, g["prob_class_1"].values)
            for fid, g in df.groupby("split")}


def compute_mean_roc(fold_dict: dict[int, tuple[np.ndarray, np.ndarray]],
                     n_points: int = 100
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calcula ROC media y desviación estándar a partir de predicciones por fold.

    Args:
        fold_dict (dict[int, tuple[np.ndarray, np.ndarray]]): Mapea fold a (y_true, y_prob).
        n_points (int): Número de puntos en grid uniforme de FPR.
    Returns:
        mean_fpr (np.ndarray): Grid uniforme de FPR en [0,1].
        mean_tpr (np.ndarray): Media de TPR interpolados en mean_fpr.
        std_tpr (np.ndarray): Desviación estándar de los TPR interpolados.
        mean_auc (float): Media de AUC por fold.
    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, n_points)

    for y_true, y_prob in fold_dict.values():
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        aucs.append(metrics.auc(fpr, tpr))
        # interpola cada ROC al grid común
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = float(np.mean(aucs))
    return mean_fpr, mean_tpr, std_tpr, mean_auc


def plot_mean_roc(dl_dict, rad_dict, out_path: Path,
                  lw_curve: float = 1.5, alpha_shading: float = 0.2):
    """
    Genera y guarda gráfico de ROC medias ± desviación para DL y Radiomics.

    Args:
        dl_dict (dict[int, tuple[np.ndarray, np.ndarray]]): Predicciones DL por fold.
        rad_dict (dict[int, tuple[np.ndarray, np.ndarray]]): Predicciones Radiomics por fold.
        out_path (Path): Ruta de salida para guardar la figura.
        lw_curve (float): Grosor de línea para las curvas.
        alpha_shading (float): Transparencia para sombreado de desviación.
    """
    # calcular medias y stds
    fpr_dl, tpr_dl_mean, tpr_dl_std, auc_dl = compute_mean_roc(dl_dict)
    fpr_r,  tpr_r_mean,  tpr_r_std,  auc_r  = compute_mean_roc(rad_dict)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Dibuja ambas curvas y guarda los handles/labels
    handle_dl, = ax.plot(fpr_dl, tpr_dl_mean,
                         label=f"config1 (AUC={auc_dl:.3f})",
                         color=COLOR_DL, lw=lw_curve)
    ax.fill_between(fpr_dl,
                    np.maximum(0, tpr_dl_mean - tpr_dl_std),
                    np.minimum(1, tpr_dl_mean + tpr_dl_std),
                    color=COLOR_DL, alpha=alpha_shading)
    handle_rad, = ax.plot(fpr_r, tpr_r_mean,
                          label=f"Regresión Logística (AUC={auc_r:.3f})",
                          color=COLOR_RAD, lw=lw_curve)
    ax.fill_between(fpr_r,
                    np.maximum(0, tpr_r_mean - tpr_r_std),
                    np.minimum(1, tpr_r_mean + tpr_r_std),
                    color=COLOR_RAD, alpha=alpha_shading)

    # Línea de azar
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

    # Etiquetas
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Ordenar la leyenda según la AUC media (de mayor a menor)
    handles_labels_aucs = [
        (handle_dl, f"config1 (AUC={auc_dl:.3f})", auc_dl),
        (handle_rad, f"Regresión Logística (AUC={auc_r:.3f})", auc_r),
    ]
    # Orden descendente
    handles_labels_aucs.sort(key=lambda x: x[2], reverse=True)

    handles = [x[0] for x in handles_labels_aucs]
    labels = [x[1] for x in handles_labels_aucs]

    leg = ax.legend(handles, labels, fontsize=10)
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(2.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   ➜ gráfico guardado en: {out_path}")

# -------------------------------- main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Curvas ROC medias ± desviación para DL vs Radiomics")
    parser.add_argument("--dl_preds_csv", type=Path, default="../../artifacts/deep_learning/gland/z_predictions/config1_predictions.csv",
                        help="CSV con predicciones del modelo DL")
    parser.add_argument("--radiomics_preds", type=Path, default="../../results/radiomics/most_discriminant/gland/preds_features_all_gland_most_discriminant.csv",
                        help="CSV 'preds_…csv' con predicciones Radiomics")
    parser.add_argument("--radiomics_model", type=str, default="Logistic Regression",
                        help="Nombre del clasificador dentro del CSV Radiomics")
    parser.add_argument("--outdir", type=Path, default=Path("../../results/compare_best_radiomics_dl"),
                        help="Directorio donde se guardarán las figuras")
    args  = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(">> Cargando Radiomics") 
    rad = load_radiomics(args.radiomics_preds, args.radiomics_model)
    print(f"   → {len(rad)} folds cargados")

    print(">> Cargando Deep Learning")
    dl = load_dl(args.dl_preds_csv)
    print(f"   → {len(dl)} folds cargados")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "roc_mean_comparison.png"

    plot_mean_roc(dl, rad, out_path)
    print("✓ Proceso completado, revisa la carpeta:", args.outdir.resolve())


if __name__ == "__main__":
    main()