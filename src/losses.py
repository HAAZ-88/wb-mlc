import numpy as np

def build_sample_weights(y, s, lambda_norm, disadvantaged_groups=[1]):
    """
    Devuelve el vector sample_weight que incorpora el término de bienestar.

    • Cada ejemplo parte con peso 1.  
    • Si pertenece a un grupo desfavorecido y su etiqueta real es positiva (y==1),
      se incrementa en λ_norm.

    Parameters
    ----------
    y : ndarray shape (N,)
        Etiquetas reales (0/1).
    s : ndarray shape (N,)
        Variable sensible (enteros 0..G-1).
    lambda_norm : float
        Peso λ_norm.
    disadvantaged_groups : list[int]
        Identificadores de grupos desfavorecidos.

    Returns
    -------
    sample_weight : ndarray shape (N,)
    """
    sample_weight = np.ones_like(y, dtype=float)
    mask = np.isin(s, disadvantaged_groups) & (y == 1)
    sample_weight[mask] += lambda_norm
    return sample_weight
