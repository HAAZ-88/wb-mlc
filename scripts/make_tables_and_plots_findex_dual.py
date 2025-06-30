# make_tables_and_plots_findex_dual.py  (versión flexible)
# --------------------------------------
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------------------------------------------
# CONFIGURACIÓN GRÁFICA
# -------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# Mapeo nombre → etiqueta legible
NOMBRES = {
    "synthetic": "Sintético",
    "bol": "Bolivia",
    "per": "Perú",
    "col": "Colombia",
    "ecu": "Ecuador",
}

RESULTS_DIR = "results"
# -------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------------
def format_row(mean, std, factor=1.0, precision=3):
    return f"{factor*mean:.{precision}f} ± {factor*std:.{precision}f}"

def resumir_tabla(df, nombre, agrupador, out_dir):
    lambdas  = sorted(df[agrupador].unique())
    filas    = []

    for lam in lambdas:
        subset = df[df[agrupador] == lam]
        fila   = [lam]
        for col in ["Accuracy", "MacroF1", "TPR_gap_squared", "DP_gap", "Welfare"]:
            mean   = subset[col].mean()
            std    = subset[col].std()
            factor = 1e8 if col == "Welfare" else 1.0
            prec   = 2   if col == "Welfare" else 3
            fila.append(format_row(mean, std, factor, prec))
        filas.append(fila)

    columnas = [r"$\lambda_{\mathrm{norm}}$", "Accuracy", "Macro-F1",
                "TPR-gap$^2$", "DP-gap", "Welfare $W^*$ ($\times 10^{-8}$)"]
    pd.DataFrame(filas, columns=columnas).to_csv(
        os.path.join(out_dir, f"table_{nombre}.csv"), index=False
    )

def graficar_metricas(df, nombre, agrupador, out_dir):
    grouped = df.groupby(agrupador)
    lambdas = sorted(grouped.groups.keys())

    def mean_ci(metric):
        m = grouped[metric].mean()
        e = 1.96 * grouped[metric].std() / len(grouped) ** 0.5
        return m, e

    #––– 1. Accuracy & Macro-F1 –––––––––––––––––––––––––––––––––
    acc_m, acc_e = mean_ci("Accuracy")
    f1_m,  f1_e  = mean_ci("MacroF1")

    plt.figure()
    plt.errorbar(lambdas, acc_m, yerr=acc_e, label="Accuracy", fmt="o-", capsize=4)
    plt.errorbar(lambdas, f1_m,  yerr=f1_e,  label="Macro-F1", fmt="s--", capsize=4)
    plt.xlabel(r"$\lambda_{\mathrm{norm}}$")
    plt.ylabel("Precisión")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_{nombre}_accuracy.png"))
    plt.close()

    #––– 2. Fairness & Welfare ––––––––––––––––––––––––––––––––––
    tpr_m, tpr_e = mean_ci("TPR_gap_squared")
    dp_m,  dp_e  = mean_ci("DP_gap")
    w_m,   w_e   = mean_ci("Welfare")

    plt.figure()
    plt.errorbar(lambdas, tpr_m, yerr=tpr_e, label="TPR-gap$^2$", fmt="o-", capsize=4)
    plt.errorbar(lambdas, dp_m,  yerr=dp_e, label="DP-gap",      fmt="s--", capsize=4)
    plt.errorbar(lambdas, w_m,   yerr=w_e,  label=r"Welfare $W^*$", fmt="^:", capsize=4)
    plt.xlabel(r"$\lambda_{\mathrm{norm}}$")
    plt.ylabel("Equidad y bienestar ($W^*$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_{nombre}_fairness_welfare.png"))
    plt.close()

def tabla_comparativa(df, dominio):
    base = df[df["lambda_norm"] == 0]  # punto de referencia
    opt  = df.copy()

    def delta(col):
        base_m = base[col].mean()
        opt_m  = opt[col].mean()
        return 100 * (opt_m - base_m) / abs(base_m)

    return {
        "Dominio": NOMBRES.get(dominio, dominio),
        r"$\Delta$ Accuracy (%)":        delta("Accuracy"),
        r"$\Delta$ Macro-F1 (%)":        delta("MacroF1"),
        "Reducción TPR-gap$^2$ (%)":    -delta("TPR_gap_squared"),
        "Reducción DP-gap (%)":         -delta("DP_gap"),
        "Mejora Welfare $W^*$ (%)":      delta("Welfare"),
    }

# -------------------------------------------------------------------
# EXPLORACIÓN AUTOMÁTICA DE ARCHIVOS
# -------------------------------------------------------------------
pat = re.compile(
    r"run_results_(?P<dom>[a-z]+)_saved_(?P<attr>[a-z]+)_grid\.csv$", re.I
)
archivos = glob.glob(os.path.join(RESULTS_DIR, "run_results_*_saved_*_grid.csv"))

if not archivos:
    raise FileNotFoundError("No se encontraron archivos *_saved_*_grid.csv en 'results/'.")

# Estructura: { atributo : { dominio : path_csv } }
atrib_universo = defaultdict(dict)
for f in archivos:
    base = os.path.basename(f)
    m = pat.match(base)
    if m:
        atrib = m.group("attr").lower()
        dom   = m.group("dom").lower()
        atrib_universo[atrib][dom] = f

# -------------------------------------------------------------------
# BUCLE PRINCIPAL POR ATRIBUTO PROTEGIDO
# -------------------------------------------------------------------
for attr, dic_dom in atrib_universo.items():
    tablas_dir = os.path.join(RESULTS_DIR, f"tablas_saved_{attr}")
    figuras_dir = os.path.join(RESULTS_DIR, f"figuras_saved_{attr}")
    os.makedirs(tablas_dir,  exist_ok=True)
    os.makedirs(figuras_dir, exist_ok=True)

    print(f"▶ Procesando atributo protegido: {attr} …")

    comparativas = []

    for dom, path_csv in dic_dom.items():
        print(f"   • {dom.upper():3}  ←  {os.path.basename(path_csv)}")

        df = pd.read_csv(path_csv)
        if "lambda_norm" not in df.columns:
            print(f"⚠️  El archivo {path_csv} no contiene la columna 'lambda_norm'. Se omite.")
            continue

        resumir_tabla(df, dom, "lambda_norm", tablas_dir)
        graficar_metricas(df, dom, "lambda_norm", figuras_dir)
        comparativas.append(tabla_comparativa(df, dom))

    #––– tabla de comparación multi-dominio para este atributo –––––
    if comparativas:
        pd.DataFrame(comparativas).round(2).to_csv(
            os.path.join(tablas_dir, "table_comparison_summary.csv"), index=False
        )

    print(f"   ✔️  Salida en: {tablas_dir}  y  {figuras_dir}\n")

print("✅ ¡Tablas y gráficas generadas para todos los atributos protegidos encontrados!")
