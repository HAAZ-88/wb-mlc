import argparse
import os
import math
import json
import glob
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy import stats

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_OUT  = os.path.join(PROJECT_DIR, "results")

# =========================
# Aliases
# =========================
COUNTRY_ALIASES = {
    "bol": "bolivia", "bolivia": "bolivia",
    "col": "colombia", "colombia": "colombia",
    "ecu": "ecuador",  "ecuador": "ecuador",
    "per": "peru",     "peru": "peru"
}

CONTEXT_ALIASES = {
    "saved_educlow": "saved_educlow",
    "saved_gender": "saved_gender",
    "saved_age": "saved_age",
    # alias mínimos
    "educlow": "saved_educlow",
    "gender": "saved_gender",
    "age": "saved_age",
}

def _code(country:str)->str:
    return {"bolivia":"bol", "colombia":"col", "ecuador":"ecu", "peru":"per"}[country]

# =========================
# Patrones de autodetección
# Nota: incluyen tu caso micro_bol_saved_age.csv
# =========================
FILENAME_PATTERNS = [
    "micro_{country}_{context}.csv",
    "micro_{countrycode}_{context}.csv",     # <== tu archivo entra aquí (micro_bol_saved_age.csv)
    "micro_{country}.csv",
    "micro_{countrycode}.csv",
    "{country}_{context}.csv",
    "{countrycode}_{context}.csv",
    "*{country}*{context}*.csv",
    "*{countrycode}*{context}*.csv",
]

def resolve_input_csv(data_dir:str, country:str, context:str, verbose:bool=True) -> str:
    """Auto-resuelve el CSV sin manifest. Muestra candidatos y aplica un scoring."""
    country = COUNTRY_ALIASES.get(country.lower(), country.lower())
    context = CONTEXT_ALIASES.get(context.lower(), context.lower())
    ccode = _code(country)

    candidates = []
    for pat in FILENAME_PATTERNS:
        pattern = pat.format(country=country, countrycode=ccode, context=context)
        full = os.path.join(data_dir, pattern)
        matches = glob.glob(full)
        candidates.extend(matches)

    # dedup case-insensitive
    seen = set(); uniq = []
    for p in candidates:
        if os.path.isfile(p):
            k = p.lower()
            if k not in seen:
                seen.add(k); uniq.append(p)

    if verbose:
        print("==> Búsqueda de CSV")
        print("   data_dir:", data_dir)
        print("   country:", country, "| context:", context, "| code:", ccode)
        print("   patrones probados:", FILENAME_PATTERNS)
        print("   candidatos:", [os.path.basename(u) for u in uniq])

    if len(uniq)==0:
        raise FileNotFoundError(
            f"No se encontró CSV para country='{country}', context='{context}' en '{data_dir}'."
        )

    # Scoring: preferimos coincidencias más “exactas”
    def score(path:str)->int:
        s = 0
        base = os.path.basename(path).lower()
        # fuerte preferencia por coincidencia exacta de patrón micro_{code}_{context}.csv
        if base == f"micro_{ccode}_{context}.csv":
            s += 100
        if base == f"micro_{country}_{context}.csv":
            s += 90
        if base.startswith("micro_"):
            s += 10
        if country in base:
            s += 4
        if ccode in base:
            s += 3
        if context in base:
            s += 4
        # penaliza archivos con 'backup' o '~'
        if "backup" in base or "~" in base:
            s -= 5
        return s

    uniq.sort(key=score, reverse=True)
    chosen = uniq[0]
    if verbose:
        print("   elegido:", os.path.basename(chosen))
    return chosen

# =========================
# Métricas / Pareto
# =========================
def tpr_by_group(y_true, y_pred, a):
    tprs = {}
    for g in np.unique(a):
        mask = (a == g) & (y_true == 1)
        denom = mask.sum()
        tprs[g] = (y_pred[mask] == 1).mean() if denom>0 else 0.0
    return tprs

def tpr_gap_sq(y_true, y_pred, a):
    tprs = tpr_by_group(y_true, y_pred, a)
    if len(tprs) < 2:
        return 0.0
    return (max(tprs.values()) - min(tprs.values()))**2

def welfare_W(y_true, p_hat, alpha=0.7, rho=2.0):
    p_hat = np.clip(p_hat, 1e-9, 1-1e-9)
    u = np.where(y_true == 1, np.power(p_hat, alpha), np.power(1.0 - p_hat, alpha))
    return np.sum(np.power(u, 1.0 - rho))

def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2: return (float(x.mean()) if n else np.nan, 0.0)
    m = x.mean(); s = x.std(ddof=1)
    h = stats.t.ppf(0.975, n-1) * s / max(1e-12, math.sqrt(n))
    return m, h

def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    n = len(points); dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]: continue
        for j in range(n):
            if i==j: continue
            if (points[j][0] >= points[i][0] and points[j][1] >= points[i][1]) and \
               (points[j][0] >  points[i][0] or  points[j][1] >  points[i][1]):
                dominated[i] = True; break
    return [i for i in range(n) if not dominated[i]]

def select_ideal(acc_list, W_list, idxs):
    acc = np.array([acc_list[i] for i in idxs], dtype=float)
    W   = np.array([W_list[i]   for i in idxs], dtype=float)
    acc_n = (acc - acc.min()) / (acc.max() - acc.min() + 1e-12)
    W_n   = (W   - W.min())   / (W.max()   - W.min()   + 1e-12)
    d = np.sqrt((1 - acc_n)**2 + (1 - W_n)**2)
    return idxs[int(np.argmin(d))]

# =========================
# Ponderación normativa
# =========================
def build_sample_weight(y_train, a_train, disadv_group, lambda_norm, base_wgt=None):
    if base_wgt is None:
        base_wgt = np.ones_like(y_train, dtype=float)
    sw = base_wgt.astype(float).copy()
    mask = (y_train == 1) & (a_train == disadv_group)
    sw[mask] = sw[mask] * (1.0 + float(lambda_norm))
    return sw

# =========================
# Pipeline por semilla
# =========================
def run_one_seed(df, target_col, protected_col, feature_cols, seed, lambdas, alpha, rho, epsilon, rf_params, weight_col):
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values
    a = df[protected_col].astype(int).values
    base_wgt = df[weight_col].values.astype(float) if (weight_col and weight_col in df.columns) else np.ones(len(df))

    # 70/15/15 estratificado por atributo protegido
    X_temp, X_test, y_temp, y_test, a_temp, a_test, w_temp, w_test = train_test_split(
        X, y, a, base_wgt, test_size=0.15, random_state=seed, stratify=a
    )
    X_train, X_val, y_train, y_val, a_train, a_val, w_train, w_val = train_test_split(
        X_temp, y_temp, a_temp, w_temp, test_size=0.1764706, random_state=seed, stratify=a_temp
    )

    results = []

    # Base para identificar grupo desaventajado
    rf0 = RandomForestClassifier(random_state=seed, **rf_params)
    rf0.fit(X_train, y_train, sample_weight=w_train)
    p_val0 = rf0.predict_proba(X_val)[:, 1]
    yhat_val0 = (p_val0 >= 0.5).astype(int)
    tprs0 = tpr_by_group(y_val, yhat_val0, a_val)
    if len(tprs0) >= 2:
        disadv_group = min(tprs0, key=tprs0.get)
    else:
        rates = {g: ((y_train[(a_train==g)]==1).mean() if (a_train==g).sum()>0 else 0.0) for g in np.unique(a_train)}
        disadv_group = min(rates, key=rates.get)

    for lam in lambdas:
        sw = build_sample_weight(y_train, a_train, disadv_group, lam, base_wgt=w_train)
        rf = RandomForestClassifier(random_state=seed, **rf_params)
        rf.fit(X_train, y_train, sample_weight=sw)

        p_val = rf.predict_proba(X_val)[:, 1]
        yhat_val = (p_val >= 0.5).astype(int)

        acc = accuracy_score(y_val, yhat_val)
        gap2 = tpr_gap_sq(y_val, yhat_val, a_val)
        W = welfare_W(y_val, p_val, alpha=alpha, rho=rho)

        results.append({
            "seed": seed,
            "lambda_norm": lam,
            "accuracy": acc,
            "tpr_gap_sq": gap2,
            "W": W,
            "disadv_group": disadv_group
        })

    res = pd.DataFrame(results)

    admiss = res[res["tpr_gap_sq"] <= epsilon].copy()
    if len(admiss) == 0:
        best_idx = int(res["tpr_gap_sq"].idxmin())
        selected = res.loc[[best_idx]].copy()
        selected["selected_with_relax"] = True
        selected["note"] = "No model passed fairness threshold; picked min gap."
        return res, selected

    points = list(zip(admiss["accuracy"].tolist(), admiss["W"].tolist()))
    pf_rel = pareto_front(points)
    pf_abs = [admiss.index.tolist()[i] for i in pf_rel]
    chosen_abs = select_ideal(res["accuracy"].tolist(), res["W"].tolist(), pf_abs)
    chosen_row = res.loc[[chosen_abs]].copy()
    chosen_row["selected_with_relax"] = False
    chosen_row["note"] = "Selected on Pareto front under fairness constraint."
    return res, chosen_row

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="WB-MLC – Random Forest (auto-resolver archivos, sin manifest)")
    parser.add_argument("--data_dir", default="data", help="Directorio con los CSV")
    parser.add_argument("--country", required=False, default="", help="bolivia/colombia/ecuador/peru (o bol,col,ecu,per)")
    parser.add_argument("--context", required=False, default="", help="saved_educlow / saved_gender / saved_age")

    parser.add_argument("--input_csv", default="", help="Ruta directa al CSV (si se da, ignora country/context)")

    parser.add_argument("--target_col", default="saved")
    parser.add_argument("--protected_col", required=True, help="educ_low / female / age_bin")
    parser.add_argument("--feature_cols", default="", help="Lista separada por comas; si vacío infiere")
    parser.add_argument("--weight_col", default="", help="p.ej. wgt; si vacío usa peso 1")

    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--lambdas", default="0,0.3,0.6,1,2,5")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--rf_params", default="", help='JSON (ej: {"n_estimators":400, "max_depth":null})')

    # (2) Cambia el default de out_dir a DEFAULT_OUT
    parser.add_argument("--out_dir", default=DEFAULT_OUT, help="Carpeta de salida (por defecto: <proyecto>/results)")
    parser.add_argument("--tag", default="", help="Etiqueta (país_contexto)")

    args = parser.parse_args()

    # (3) Normaliza y crea la carpeta de salida
    if not os.path.isabs(args.out_dir):
        # si alguien pasa un out_dir relativo, lo hacemos relativo al PROYECTO (no al cwd)
        args.out_dir = os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    

    # === Resolución del CSV ===
    if args.input_csv.strip():
        csv_path = args.input_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(args.data_dir, csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"--input_csv apunta a un archivo inexistente: {csv_path}")
        print("==> CSV (input_csv):", csv_path)
    else:
        if not args.country or not args.context:
            raise ValueError("Si no usas --input_csv, debes proporcionar --country y --context.")
        csv_path = resolve_input_csv(args.data_dir, args.country, args.context, verbose=True)

    df = pd.read_csv(csv_path)

    # Features por defecto
    if args.feature_cols.strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        drop_cols = {args.target_col, args.protected_col}
        bad_like = {"economy", "economycode", "wpid_random"}
        feature_cols = [c for c in df.columns if c not in drop_cols and c not in bad_like]
        if args.weight_col and args.weight_col in feature_cols:
            feature_cols.remove(args.weight_col)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.rf_params.strip():
        rf_params = json.loads(args.rf_params)
        for k,v in list(rf_params.items()):
            if isinstance(v, str) and v.lower()=="null":
                rf_params[k]=None
    else:
        rf_params = dict(
            n_estimators=400, max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            n_jobs=-1, class_weight=None
        )

    all_rows = []; selected_rows = []

    # Etiqueta por defecto si no se pasa --tag
    if args.tag:
        tag = args.tag
    else:
        # si hubo autodetección, construimos tag país_contexto; si hubo --input_csv, usamos basename
        if args.input_csv.strip():
            tag = os.path.splitext(os.path.basename(csv_path))[0]
        else:
            ctry = COUNTRY_ALIASES.get(args.country.lower(), args.country.lower())
            ctx  = CONTEXT_ALIASES.get(args.context.lower(), args.context.lower())
            tag = f"{ctry}_{ctx}"

    for seed in seeds:
        res, chosen = run_one_seed(
            df=df,
            target_col=args.target_col,
            protected_col=args.protected_col,
            feature_cols=feature_cols,
            seed=seed,
            lambdas=lambdas,
            alpha=args.alpha, rho=args.rho,
            epsilon=args.epsilon,
            rf_params=rf_params,
            weight_col=args.weight_col if args.weight_col else None
        )
        res["seed"] = seed; res["tag"] = tag
        chosen["seed"] = seed; chosen["tag"] = tag
        all_rows.append(res); selected_rows.append(chosen)

    all_df = pd.concat(all_rows, ignore_index=True)
    sel_df = pd.concat(selected_rows, ignore_index=True)

    acc_mean, acc_ci = ci95(sel_df["accuracy"].values)
    gap_mean, gap_ci = ci95(sel_df["tpr_gap_sq"].values)
    W_mean, W_ci = ci95(sel_df["W"].values)

    summary = pd.DataFrame([{
        "tag": tag,
        "n_seeds": len(seeds),
        "accuracy_mean": acc_mean, "accuracy_ci95": acc_ci,
        "tpr_gap_sq_mean": gap_mean, "tpr_gap_sq_ci95": gap_ci,
        "W_mean": W_mean, "W_ci95": W_ci,
        "lambda_modal": sel_df["lambda_norm"].mode().iloc[0] if len(sel_df)>0 else np.nan,
        "selected_with_relax_any": bool((sel_df.get("selected_with_relax", False)==True).any()),
        "csv_used": csv_path,
        "features_used": ";".join(feature_cols)
    }])

    base = tag
    all_df.to_csv(os.path.join(args.out_dir, f"{base}_rf_all.csv"), index=False)
    sel_df.to_csv(os.path.join(args.out_dir, f"{base}_rf_selected_models.csv"), index=False)

    summary_path = os.path.join(args.out_dir, "rf_summary.csv")
    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        out = pd.concat([prev, summary], ignore_index=True)
    else:
        out = summary
    out.to_csv(summary_path, index=False)

    print("==> Guardado:")
    print(" - Todos los resultados:", os.path.join(args.out_dir, f"{base}_rf_all.csv"))
    print(" - Seleccionados:", os.path.join(args.out_dir, f"{base}_rf_selected_models.csv"))
    print(" - Resumen acumulado:", summary_path)
    print(" - Features usadas:", feature_cols)

if __name__ == "__main__":
    main()