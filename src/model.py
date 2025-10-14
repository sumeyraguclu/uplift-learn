# ==================== src/model.py ====================
"""
T-Learner ve diğer uplift modelleri

Bu modül Gün 2'de dolduracağız.
"""

import numpy as np
from typing import Any


class TLearner:
    """
    T-Learner (Two-Model) uplift modeling
    
    İki ayrı model eğitir:
    - model_treatment: Treatment grubunda P(Y=1|X)
    - model_control: Control grubunda P(Y=1|X)
    
    Uplift = model_treatment(X) - model_control(X)
    
    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> model = TLearner(
    ...     treatment_estimator=XGBClassifier(),
    ...     control_estimator=XGBClassifier()
    ... )
    >>> model.fit(X_train, y_train, treatment_train)
    >>> uplift = model.predict(X_test)
    """
    
    def __init__(self, treatment_estimator: Any, control_estimator: Any):
        """
        Parameters
        ----------
        treatment_estimator : estimator
            Treatment grubu için model (scikit-learn API)
        control_estimator : estimator
            Control grubu için model (scikit-learn API)
        """
        # TODO: Gün 2'de implement et
        raise NotImplementedError("Bu sınıf henüz implement edilmedi")
    
    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray):
        """
        İki modeli eğit
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Özellikler
        y : array-like, shape (n_samples,)
            Hedef değişken (0 veya 1)
        treatment : array-like, shape (n_samples,)
            Treatment göstergesi (0 veya 1)
        """
        # TODO: Gün 2'de implement et
        raise NotImplementedError()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uplift tahmin et
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Özellikler
        
        Returns
        -------
        uplift : array-like, shape (n_samples,)
            Uplift skorları (p1 - p0)
        """
        # TODO: Gün 2'de implement et
        raise NotImplementedError()
