import numpy as np

# ---------------------------------------------------------------------
# Brechas de equidad (sin cambios)
# ---------------------------------------------------------------------
def tpr_gap(y_true, y_pred, s):
    ref = (y_pred[y_true == 1] == 1).mean()
    gaps = []
    for g in np.unique(s):
        mask = (s == g) & (y_true == 1)
        tpr_g = (y_pred[mask] == 1).mean() if mask.any() else 0.0
        gaps.append(abs(tpr_g - ref))
    return max(gaps)

def dp_gap(y_pred, s):
    ref = (y_pred == 1).mean()
    gaps = []
    for g in np.unique(s):
        dp = (y_pred[s == g] == 1).mean()
        gaps.append(abs(dp - ref))
    return max(gaps)

# ---------------------------------------------------------------------
# Utilidad continua prioritarista
# ---------------------------------------------------------------------
def _prioritarian_utility(y_true, y_prob, alpha=0.5):
    """
    Devuelve u_i como función prioritarista continua de la probabilidad predictiva.
    Por defecto, usa exponente α = 0.5, más sensible a predicciones inciertas.
    """
    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    u = np.where(y_true == 1,
                 np.power(p, alpha),
                 np.power(1 - p, alpha))
    return u

# ---------------------------------------------------------------------
# Índice de bienestar prioritarista continuo
# ---------------------------------------------------------------------
def welfare_index(y_true,
                  y_prob,
                  rho=2.0,
                  alpha=0.5):
    """
    W* = -(1 - rho) * sum_i [ u_i^{1 - rho} / (1 - rho) ]
       = sum_i [ u_i^{1 - rho} ]         si se reescala

    Parámetros:
    - y_true : etiquetas reales (0 o 1)
    - y_prob : probabilidad predicha de la clase 1
    - rho    : grado de aversión a la desigualdad (ρ > 1 para prioritarismo)
    - alpha  : suavización de la utilidad predictiva (por defecto 0.5)
    """
    u = _prioritarian_utility(y_true, y_prob, alpha=alpha)
    if rho == 1:
        return np.exp(np.mean(np.log(u)))  # media geométrica (caso límite)
    else:
        return np.sum(u ** (1 - rho))
