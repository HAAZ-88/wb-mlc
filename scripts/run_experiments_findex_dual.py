"""
run_experiments_findex_dual.py
---------------------------------
Versión actualizada para evaluar WB‑ML en datos Global Findex con:
      data/micro_bol_saved_educlow.csv ➜ bol_saved_educlow
"""

import argparse
import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Importar modelo y métricas desde src (asumiendo estructura de proyecto)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import WBLogisticModel  # type: ignore
from src.metrics import tpr_gap, dp_gap, welfare_index  # type: ignore


# --------------------------------------------------
def tpr_gap_squared(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    """Brecha de TPR al cuadrado (penaliza más las diferencias)."""
    return tpr_gap(y_true, y_pred, s) ** 2


# --------------------------------------------------
def split_data(
    df: pd.DataFrame, label_col: str, prot_col: str, seed: int
) -> Tuple[np.ndarray, ...]:
    """Divide el dataset en train/val/test conservando proporciones de la clase."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = num_cols.drop([label_col, prot_col])
    X = df[feature_cols].values
    y = df[label_col].values
    s = df[prot_col].values

    X_tr, X_temp, y_tr, y_temp, s_tr, s_temp = train_test_split(
        X, y, s, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_te, y_val, y_te, s_val, s_te = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return (
        X_tr,
        y_tr,
        s_tr,
        X_val,
        y_val,
        s_val,
        X_te,
        y_te,
        s_te,
    )


# --------------------------------------------------
def evaluate(
    model: WBLogisticModel,
    X: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    rho: float,
    alpha: float,
) -> dict:
    """Calcula métricas básicas y prioritaristas para un conjunto dado."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "MacroF1": f1_score(y, y_pred, average="macro"),
        "TPR_gap_squared": tpr_gap_squared(y, y_pred, s),
        "DP_gap": dp_gap(y_pred, s),
        "Welfare": welfare_index(y, y_prob, rho=rho, alpha=alpha),
    }


# --------------------------------------------------
def build_tag_from_path(path: str) -> str:
    """Extrae un tag descriptivo del nombre del archivo de datos.

    Ejemplo:
        data/micro_bol_saved_educlow.csv -> bol_saved_educlow
    """
    base = os.path.splitext(os.path.basename(path))[0]  # micro_bol_saved_educlow
    parts = base.split("_")
    return "_".join(parts[1:]) if len(parts) > 1 else base


# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta grid‑search de WB‑ML y guarda métricas en CSV."
    )
    parser.add_argument("--data_path", required=True, help="Ruta al CSV de entrada")
    parser.add_argument(
        "--lambda_grid",
        nargs="+",
        type=float,
        default=[0, 0.3, 0.6, 1, 2, 5],
        help="Valores de λ_norm a evaluar",
    )
    parser.add_argument("--n_seeds", type=int, default=10, help="Número de semillas")
    parser.add_argument("--label_col", required=True, help="Columna objetivo (y)")
    parser.add_argument(
        "--protected_col", required=True, help="Columna de atributo protegido (s)"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=2.0,
        help="Aversión a la desigualdad (rho>1 => prioritarista)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Exponente alpha de la utilidad prioritarista continua",
    )
    args = parser.parse_args()

    # Preparación
    df = pd.read_csv(args.data_path)
    os.makedirs("results", exist_ok=True)
    tag = build_tag_from_path(args.data_path)

    grid: List[dict] = []
    best: List[dict] = []

    # ------------------------- Experimentos -------------------------
    for seed in range(args.n_seeds):
        (
            X_tr,
            y_tr,
            s_tr,
            X_val,
            y_val,
            s_val,
            X_te,
            y_te,
            s_te,
        ) = split_data(df, args.label_col, args.protected_col, seed)

        scores = {}
        models = {}

        for lam in args.lambda_grid:
            mdl = WBLogisticModel(lambda_norm=lam, C=0.2)
            mdl.fit(X_tr, y_tr, s_tr)

            # Validación
            val_m = evaluate(mdl, X_val, y_val, s_val, args.rho, args.alpha)
            # Función objetivo (ajusta pesos si lo deseas)
            score = val_m["Accuracy"] - 2 * val_m["TPR_gap_squared"] + 0.2 * val_m["Welfare"]
            scores[lam] = score
            models[lam] = mdl

            # Test – guardamos todas las combinaciones
            test_m = evaluate(mdl, X_te, y_te, s_te, args.rho, args.alpha)
            test_m.update(seed=seed, lambda_norm=lam)
            grid.append(test_m)

        # Mejor λ según validación
        best_lam = max(scores, key=scores.get)
        best_m = evaluate(models[best_lam], X_te, y_te, s_te, args.rho, args.alpha)
        best_m.update(seed=seed, lambda_hat=best_lam)
        best.append(best_m)

    # ------------------------- Guardar resultados -------------------------
    grid_path = f"results/run_results_{tag}_grid.csv"
    best_path = f"results/run_results_{tag}.csv"
    pd.DataFrame(grid).to_csv(grid_path, index=False)
    pd.DataFrame(best).to_csv(best_path, index=False)
    print(f"✅ Resultados guardados en:\n   • {grid_path}\n   • {best_path}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
