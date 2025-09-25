# scripts/shap_explain.py
import argparse, json
from pathlib import Path
import joblib, shap, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_meta(meta_path):
    meta = json.load(open(meta_path))
    df = pd.read_csv(meta["data_csv"])
    X = df[meta["features"]].copy()
    y = df["saved"].astype(int).values
    a = df[meta["protected"]].astype(int).values
    return meta, X, y, a

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_group_violin(shap_values, a, out_png):
    import seaborn as sns
    plt.figure(figsize=(7,4))
    df = pd.DataFrame({"abs_shap": np.abs(shap_values).mean(axis=1), "group": a})
    sns.violinplot(x="group", y="abs_shap", data=df, cut=0)
    plt.title("Distribución |SHAP| por grupo protegido")
    plt.xlabel("grupo (0=no protegido, 1=protegido)")
    plt.ylabel("contribución media absoluta")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True, help="ruta base sin sufijo (.joblib/.json)")
    ap.add_argument("--out_dir", default="artifacts/shap")
    ap.add_argument("--subset", choices=["train","test","all"], default="all")
    ap.add_argument("--topk_dep", type=int, default=3, help="número de features para dependence plots")
    args = ap.parse_args()

    base = Path(args.model_base)
    mdl = joblib.load(base.with_suffix(".joblib"))
    meta, X, y, a = load_meta(base.with_suffix(".json"))
    ensure_dir(Path(args.out_dir))

    # Para simplicidad usamos TODO el dataset; si prefieres valid/test, añade índices guardados en Fase 1.
    X_use, y_use, a_use = X, y, a

    # Elegir explainer según modelo
    if hasattr(mdl, "predict_proba") and hasattr(mdl, "coef_"):
        explainer = shap.Explainer(mdl, X_use)
        sv = explainer(X_use)       # shap.Explanation
        shap_values = sv.values     # para mantener compatibilidad con el resto del script
    else:
        explainer = shap.TreeExplainer(mdl)
        shap_values = explainer.shap_values(X_use)

    # Summary (beeswarm)
    out_sum = Path(args.out_dir)/f"{base.name}_summary.png"
    shap.summary_plot(shap_values, X_use, show=False)
    plt.tight_layout(); plt.savefig(out_sum, dpi=200); plt.close()

    # Summary bar (importancias globales)
    out_bar = Path(args.out_dir)/f"{base.name}_bar.png"
    shap.summary_plot(shap_values, X_use, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(out_bar, dpi=200); plt.close()

    # Dependence plots de las top-k features globales
    abs_mean = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(abs_mean)[::-1][:args.topk_dep]
    for j in top_idx:
        fname = X_use.columns[j]
        out_dep = Path(args.out_dir)/f"{base.name}_dep_{fname}.png"
        shap.dependence_plot(j, shap_values, X_use, show=False)
        plt.tight_layout(); plt.savefig(out_dep, dpi=200); plt.close()

    # Violin por grupo (¿hay sistematicidad de contribuciones absolutas?)
    out_vio = Path(args.out_dir)/f"{base.name}_group_violin.png"
    plot_group_violin(shap_values, a_use, out_vio)

    # Opcional: explicar casos representativos (TP, FP, FN)
    p = mdl.predict_proba(X_use)[:,1]
    yhat = (p>=0.5).astype(int)
    mask_tp = (yhat==1) & (y_use==1)
    mask_fp = (yhat==1) & (y_use==0)
    mask_fn = (yhat==0) & (y_use==1)

    for label,mask in [("TP",mask_tp),("FP",mask_fp),("FN",mask_fn)]:
        idxs = np.where(mask)[0][:3]  # hasta 3 casos
        for k,ix in enumerate(idxs):
            out_force = Path(args.out_dir)/f"{base.name}_force_{label}_{k}.png"
            try:
                # Force plot estático: usa matplotlib backend
                fp = shap.force_plot(explainer.expected_value, shap_values[ix,:], X_use.iloc[ix,:], matplotlib=True, show=False)
                plt.tight_layout(); plt.savefig(out_force, dpi=200); plt.close()
            except Exception:
                # En entornos sin compatibilidad, generar waterfall alternativo
                out_water = Path(args.out_dir)/f"{base.name}_waterfall_{label}_{k}.png"
                shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[ix,:], feature_names=X_use.columns, show=False)
                plt.tight_layout(); plt.savefig(out_water, dpi=200); plt.close()

    # Guardar un pequeño reporte CSV con importancias globales
    imp = pd.DataFrame({"feature": X_use.columns, "abs_mean_shap": abs_mean})
    imp.sort_values("abs_mean_shap", ascending=False, inplace=True)
    imp.to_csv(Path(args.out_dir)/f"{base.name}_global_importance.csv", index=False)

    print(f"[OK] SHAP listo para {base.name}. Salidas en {args.out_dir}")

if __name__ == "__main__":
    main()