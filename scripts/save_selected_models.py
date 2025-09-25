# scripts/save_selected_models.py
# Guarda los modelos finales (y metadatos) para usar luego con SHAP
import argparse, json, os, joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# === CONFIG BÁSICA: ajusta si tus nombres varían ===
CONTEXT_MAP = {
    "saved_educlow": {"protected": "educ_low"},
    "saved_gender":  {"protected": "female_bin"},
    "saved_age":     {"protected": "age_bin"},
}

# Los predictores por contexto (ajusta si tu CSV ya viene filtrado)
PREDICTORS = {
    "saved_educlow": ["female","age","educ","emp_in","mobileowner"],
    "saved_gender":  ["age","emp_in","mobileowner","internetaccess","anydigpayment"],
    "saved_age":     ["educ","emp_in","mobileowner","internetaccess","anydigpayment"],
}

LAMBDA_GRID = [0.0, 0.3, 0.6, 1.0, 2.0, 5.0]
EPS = 1e-3
SEEDS = list(range(10))  # 10 semillas como en el paper

def make_weights(y, a, base_w=1.0, lambda_norm=0.0):
    """
    Ponderación prioritarista mínima: incrementa peso de positivos (y=1) del grupo con menor TPR.
    Nota: Para seleccionar el modelo final, necesitas calcular TPR por grupo primero (con pesos base).
    Aquí usamos una aproximación consistente con tu 3.3: una vez identificado el grupo desfavorecido,
    se aumenta su peso en positivos en proporción a lambda_norm.
    """
    import numpy as np
    w = np.ones_like(y, dtype=float) * base_w
    # Calcular TPR por grupo con pesos base
    tpr = {}
    for g in [0,1]:
        mask_pos = (y==1) & (a==g)
        denom = mask_pos.sum()
        if denom == 0:
            tpr[g] = 0.0
        else:
            # proxy rápido: sin modelo aún, usa tasa base de positivos por grupo (sirve para decidir "desaventajado")
            tpr[g] = denom / (a==g).sum()
    disadvantaged = 0 if tpr[0] < tpr[1] else 1
    # Aumenta peso de positivos del grupo desfavorecido
    w[(y==1) & (a==disadvantaged)] *= (1.0 + lambda_norm)
    return w, disadvantaged

def tpr_gap2(y_true, y_pred, a):
    import numpy as np
    tpr = []
    for g in [0,1]:
        mask = (a==g) & (y_true==1)
        denom = mask.sum()
        if denom == 0:
            tpr_g = 0.0
        else:
            tpr_g = (y_pred[mask]==1).mean()
        tpr.append(tpr_g)
    gap = (tpr[1]-tpr[0])**2
    return gap

def welfare_index(y_true, p_hat, alpha=0.7, rho=2.0):
    """
    Implementa tu W simple: u_i = p_i^alpha si y=1; (1-p_i)^alpha si y=0; W = sum(u_i^(1-rho))
    """
    import numpy as np
    p = p_hat.clip(1e-6, 1-1e-6)
    u = np.where(y_true==1, p**alpha, (1-p)**alpha)
    return np.sum(u**(1.0-rho))

def fit_logit(X, y, sample_weight=None):
    m = LogisticRegression(max_iter=1000, solver="lbfgs")
    m.fit(X, y, sample_weight=sample_weight)
    return m

def select_model_on_grid(X_tr, y_tr, a_tr, X_val, y_val, a_val):
    """
    Aplica tu regla: (i) filtra por TPR-gap^2 <= EPS, (ii) frontera (Accuracy, W), (iii) distancia al ideal.
    Devuelve el mejor modelo y el lambda seleccionado.
    """
    import numpy as np
    cand = []
    for lam in LAMBDA_GRID:
        w_tr, _ = make_weights(y_tr, a_tr, lambda_norm=lam)
        mdl = fit_logit(X_tr, y_tr, sample_weight=w_tr)
        p_val = mdl.predict_proba(X_val)[:,1]
        y_pred = (p_val>=0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        gap2 = tpr_gap2(y_val, y_pred, a_val)
        W = welfare_index(y_val, p_val)
        cand.append({"lam":lam,"mdl":mdl,"acc":acc,"gap2":gap2,"W":W})
    # filtrar por equidad mínima
    feas = [c for c in cand if c["gap2"] <= EPS]
    if not feas:
        # si no hay factibles, devolver el de menor gap (documentado en 4.7)
        feas = [min(cand, key=lambda c: c["gap2"])]
    # frontera de Pareto en (acc, W)
    A = [(c["acc"], c["W"]) for c in feas]
    # no dominados
    pareto = []
    for i,ci in enumerate(feas):
        ai = A[i]
        dominated = False
        for j,cj in enumerate(feas):
            if j==i: continue
            aj = A[j]
            if (aj[0] > ai[0]) and (aj[1] > ai[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(ci)
    # distancia al ideal (normalizando)
    accs = [c["acc"] for c in pareto]
    Ws   = [c["W"]   for c in pareto]
    acc_min, acc_max = min(accs), max(accs)
    W_min,   W_max   = min(Ws),   max(Ws)
    def norm(x,xmin,xmax): 
        return 0.0 if xmax==xmin else (x - xmin)/(xmax - xmin)
    best = None; bestd = 1e9
    for c in pareto:
        acc_n = norm(c["acc"], acc_min, acc_max)
        W_n   = norm(c["W"],   W_min,   W_max)
        d = ((1-acc_n)**2 + (1-W_n)**2)**0.5
        if d < bestd:
            bestd = d; best = c
    return best["mdl"], best["lam"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)  # data/micro_bolivia_saved_gender.csv
    ap.add_argument("--context", required=True, choices=list(CONTEXT_MAP.keys()))
    ap.add_argument("--country", required=True)
    ap.add_argument("--out_dir", default="artifacts/models")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    protected = CONTEXT_MAP[args.context]["protected"]
    feats = PREDICTORS[args.context]  # o list(df.columns) si ya viene filtrado
    X = df[feats].copy()
    y = df["saved"].astype(int).values
    a = df[protected].astype(int).values

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    records = []

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.30, random_state=123)  # 70/30 (y adentro 15/15 puedes dividir)
    fold = 0
    for tr_idx, te_idx in sss.split(X, a):  # estratifica por atributo protegido
        fold += 1
        X_tr0, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr0, y_te = y[tr_idx], y[te_idx]
        a_tr0, a_te = a[tr_idx], a[te_idx]

        # dividir tr0 en train/val (85/15 aprox para replicar 70/15/15 global)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=fold)  # ~15% de 85% ≈ 15/100
        tr_idx2, va_idx = next(sss2.split(X_tr0, a_tr0))
        X_tr, X_val = X_tr0.iloc[tr_idx2], X_tr0.iloc[va_idx]
        y_tr, y_val = y_tr0[tr_idx2], y_tr0[va_idx]
        a_tr, a_val = a_tr0[tr_idx2], a_tr0[va_idx]

        mdl, lam = select_model_on_grid(X_tr, y_tr, a_tr, X_val, y_val, a_val)

        # guarda artefactos
        tag = f"{args.country}_{args.context}_seed{fold}"
        base = Path(args.out_dir)/tag
        base.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(mdl, base.with_suffix(".joblib"))
        meta = {
            "country": args.country,
            "context": args.context,
            "seed": fold,
            "lambda_norm": lam,
            "protected": protected,
            "features": feats,
            "data_csv": args.data_csv
        }
        json.dump(meta, open(base.with_suffix(".json"), "w"), indent=2)
        records.append(meta)

    # índice general
    out_index = Path(args.out_dir)/f"index_{args.country}_{args.context}.json"
    json.dump(records, open(out_index, "w"), indent=2)
    print(f"[OK] Modelos guardados e indexados en: {out_index}")

if __name__ == "__main__":
    main()