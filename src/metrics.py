# ==================== src/metrics.py ====================
"""
Uplift modeling metrikleri

Bu modül Gün 3'te dolduracağız.
"""

import numpy as np


def qini_auc_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray
) -> float:
    """
    Qini AUC (Area Under Qini Curve)
    
    Uplift modelinin performans metriği.
    Yüksek değer = İyi model
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Gerçek outcome (0 veya 1)
    uplift : array-like, shape (n_samples,)
        Model uplift tahminleri
    treatment : array-like, shape (n_samples,)
        Treatment göstergesi (0 veya 1)
    
    Returns
    -------
    qini_auc : float
        Qini AUC skoru
    
    Examples
    --------
    >>> qini = qini_auc_score(y_test, uplift_pred, treatment_test)
    >>> print(f"Qini AUC: {qini:.4f}")
    """
    # TODO: Gün 3'te implement et
    raise NotImplementedError("Bu fonksiyon henüz implement edilmedi")


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.3
) -> float:
    """
    Uplift@k: İlk k% müşteride ortalama uplift
    
    Parameters
    ----------
    y_true : array-like
        Gerçek outcome
    uplift : array-like
        Model uplift tahminleri
    treatment : array-like
        Treatment göstergesi
    k : float
        Hedeflenen yüzde (0-1 arası)
    
    Returns
    -------
    uplift_k : float
        İlk k%'deki ortalama uplift
    """
    # TODO: Gün 3'te implement et
    raise NotImplementedError("Bu fonksiyon henüz implement edilmedi")

