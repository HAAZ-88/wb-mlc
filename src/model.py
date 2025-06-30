import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# Intentar importar la función original; si no existe, usar un stub
# ---------------------------------------------------------------------
try:
    from src.losses import build_sample_weights        # versión completa
except (ImportError, ModuleNotFoundError):
    def build_sample_weights(y, s, lambda_norm=0.0,
                             disadvantaged_groups=(1,)):
        """Fallback: pesos uniformes cuando falta la implementación."""
        return np.ones_like(y, dtype=float)

# ---------------------------------------------------------------------
class WBLogisticModel:
    """
    Regresión logística con ponderación de bienestar e imputación de NaN.

    Parameters
    ----------
    lambda_norm : float
        Peso del término normativo (λ_norm) usado en build_sample_weights.
    disadvantaged_groups : list[int] | tuple[int]
        Grupos considerados desfavorecidos (por defecto [1]).
    C : float
        Parámetro de regularización L2 de LogisticRegression.
    max_iter : int
        Número máximo de iteraciones del optimizador.
    """
    def __init__(self,
                 lambda_norm: float = 0.0,
                 disadvantaged_groups=(1,),
                 C: float = 1.0,
                 max_iter: int = 1000):
        self.lambda_norm = lambda_norm
        self.disadvantaged_groups = disadvantaged_groups

        # --- Pipeline: imputación → escalado → regresión logística ---
        self.pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler(with_mean=False)),
            ("clf",    LogisticRegression(
                           C=C,
                           max_iter=max_iter,
                           solver="lbfgs",
                           penalty="l2"))
        ])

    # -----------------------------------------------------------------
    def fit(self, X, y, s):
        """
        Entrena la regresión logística ponderada.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        s : array-like, shape (n_samples,)
            Atributo protegido.
        """
        sample_w = build_sample_weights(
            y, s,
            lambda_norm=self.lambda_norm,
            disadvantaged_groups=self.disadvantaged_groups
        )
        # Paso de sample_weight a la última etapa del pipeline
        self.pipeline.fit(X, y, clf__sample_weight=sample_w)

    # -----------------------------------------------------------------
    def predict(self, X):
        """Devuelve la predicción binaria."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Devuelve la probabilidad de la clase positiva."""
        return self.pipeline.predict_proba(X)[:, 1]
