
#!/usr/bin/env python3
"""Genera resumen y figura de resultados óptimos por país y contexto.

Este script asume que los archivos *_opt_models_*.csv y *_grid.csv están en una carpeta
llamada "results" y tienen nombres estructurados como:
    <país>_opt_models_<contexto>.csv
    run_results_<país>_<contexto>_grid.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def cargar_resultados_optimos(folder):
    opt_models = []
    for fname in os.listdir(folder):
        if "_opt_models_" in fname and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            base = fname.replace(".csv", "")
            if "_opt_models_" in base:
                parts = base.split("_opt_models_")
                df["país"] = parts[0]
                df["contexto"] = parts[1]
            else:
                df["país"] = base
                df["contexto"] = "default"
            opt_models.append(df)
    return pd.concat(opt_models, ignore_index=True)

def generar_resumen(df, outpath):
    resumen = df.groupby(["país", "contexto"]).agg({
        "Accuracy": ["mean", "std"],
        "TPR_gap_squared": ["mean", "std"],
        "Welfare": ["mean", "std"],
        "lambda_norm": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    }).round(4)
    resumen.columns = ["_".join(col).strip() for col in resumen.columns.values]
    resumen.reset_index(inplace=True)
    resumen.to_csv(outpath, index=False)
    print(f"Resumen guardado en {outpath}")

def generar_figuras_pareto(folder, df_opt, outdir):
    os.makedirs(outdir, exist_ok=True)
    for _, row in df_opt.iterrows():
        pais = row["país"]
        contexto = row["contexto"]
        acc_sel = row["Accuracy"]
        w_sel = row["Welfare"]
        fname_grid = f"run_results_{pais}_{contexto}_grid.csv"
        path_grid = os.path.join(folder, fname_grid)
        if not os.path.exists(path_grid):
            print(f"Archivo no encontrado: {fname_grid}")
            continue
        df_grid = pd.read_csv(path_grid)
        plt.figure()
        sns.scatterplot(data=df_grid, x="Accuracy", y="Welfare", alpha=0.3, label="Todos los modelos")
        plt.scatter(acc_sel, w_sel, color="red", label="Modelo ideal")
        plt.title(f"{pais} - {contexto}")
        plt.xlabel("Accuracy")
        plt.ylabel("Welfare")
        plt.legend()
        outfig = os.path.join(outdir, f"fig_pareto_{pais}_{contexto}.png")
        plt.savefig(outfig)
        plt.close()
        print(f"Figura guardada: {outfig}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Carpeta con los archivos CSV de resultados")
    parser.add_argument("--out_csv", default="resumen_opt_models.csv", help="Nombre del archivo de salida con el resumen")
    parser.add_argument("--figures_dir", default="figures", help="Carpeta donde guardar las figuras")
    args = parser.parse_args()

    df_opt = cargar_resultados_optimos(args.results_dir)
    generar_resumen(df_opt, args.out_csv)
    generar_figuras_pareto(args.results_dir, df_opt, args.figures_dir)

if __name__ == "__main__":
    main()
