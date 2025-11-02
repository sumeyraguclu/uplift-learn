# ==================== src/model.py ====================
"""
T-Learner ve diğer uplift modelleri

Production-grade T-Learner (Two-Model Learner) implementation
Reference: https://github.com/maks-sh/scikit-uplift
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class TLearner:
    """
    T-Learner (Two-Model) uplift modeling
    
    İki ayrı model eğitir:
    - model_treatment: Treatment grubunda P(Y=1|X)
    - model_control: Control grubunda P(Y=1|X)
    
    CATE = P(Y=1|X,T=1) - P(Y=1|X,T=0)
    
    Theory:
    -------
    - mu_0(X): E[Y|X, T=0]  (control outcome model)
    - mu_1(X): E[Y|X, T=1]  (treatment outcome model)
    - tau(X) = mu_1(X) - mu_0(X)  (conditional average treatment effect)
    
    Advantages:
    -----------
    - Separate models → capture group-specific patterns
    - Less bias than S-Learner
    - Interpretable (can understand each group separately)
    - Industry standard for uplift modeling
    
    Parameters
    ----------
    treatment_estimator : estimator, optional
        Treatment grubu için model (scikit-learn API).
        Default: XGBClassifier with standard params
    control_estimator : estimator, optional
        Control grubu için model (scikit-learn API).
        Default: XGBClassifier with standard params
    random_state : int, optional
        Random seed for reproducibility. Default: 42
    
    Attributes
    ----------
    model_0 : estimator
        Fitted control model
    model_1 : estimator
        Fitted treatment model
    scaler : StandardScaler
        Feature scaler
    feature_cols : list
        Feature column names
    is_fitted : bool
        Whether model has been fitted
    
    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.model import TLearner
    >>> 
    >>> # Basic usage with defaults
    >>> model = TLearner(random_state=42)
    >>> model.fit(X_train, y_train, treatment_train)
    >>> predictions = model.predict_cate(X_test)
    >>> 
    >>> # Custom estimators
    >>> model = TLearner(
    ...     treatment_estimator=XGBClassifier(max_depth=3),
    ...     control_estimator=XGBClassifier(max_depth=3),
    ...     random_state=42
    ... )
    >>> model.fit(X_train, y_train, treatment_train)
    >>> 
    >>> # Save/load
    >>> model.save('models/my_tlearner.pkl')
    >>> loaded_model = TLearner.load('models/my_tlearner.pkl')
    """
    
    def __init__(
        self,
        treatment_estimator: Optional[Any] = None,
        control_estimator: Optional[Any] = None,
        random_state: int = 42
    ):
        """Initialize T-Learner with estimators"""
        self.random_state = random_state
        
        # Default estimators: XGBoost with reasonable params
        if treatment_estimator is None:
            treatment_estimator = XGBClassifier(
                max_depth=5,
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state,
                verbosity=0
            )
        
        if control_estimator is None:
            control_estimator = XGBClassifier(
                max_depth=5,
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state,
                verbosity=0
            )
        
        self.treatment_estimator = treatment_estimator
        self.control_estimator = control_estimator
        
        # Will be set during fit
        self.model_0 = None
        self.model_1 = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
        self._train_metrics = {}
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Fit T-Learner models
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Features
        y : array-like, shape (n_samples,)
            Target variable (0 or 1)
        treatment : array-like, shape (n_samples,)
            Treatment indicator (0 or 1)
        test_size : float, optional
            Proportion for test set. Default: 0.2
        verbose : bool, optional
            Print training progress. Default: True
        
        Returns
        -------
        metrics : dict
            Training metrics including:
            - 'auc_0': Control model AUC
            - 'auc_1': Treatment model AUC
            - 'n_train': Training samples
            - 'n_test': Test samples
            - 'n_control': Control samples
            - 'n_treatment': Treatment samples
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_cols = X.columns.tolist()
            X = X.values
        else:
            self.feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
        
        y = np.asarray(y)
        treatment = np.asarray(treatment)
        
        if verbose:
            print("=" * 70)
            print("T-LEARNER TRAINING")
            print("=" * 70)
            print(f"Features: {len(self.feature_cols)}")
            print(f"Samples: {len(X):,}")
            print(f"Treatment: {(treatment==1).sum():,} | Control: {(treatment==0).sum():,}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
            X_scaled, y, treatment,
            test_size=test_size,
            random_state=self.random_state,
            stratify=treatment
        )
        
        if verbose:
            print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Split by treatment group
        mask_control_train = T_train == 0
        mask_treatment_train = T_train == 1
        
        X_control_train = X_train[mask_control_train]
        y_control_train = y_train[mask_control_train]
        
        X_treatment_train = X_train[mask_treatment_train]
        y_treatment_train = y_train[mask_treatment_train]
        
        # Fit control model
        if verbose:
            print(f"\nFitting control model (n={len(X_control_train):,})...")
        
        self.model_0 = self.control_estimator
        self.model_0.fit(X_control_train, y_control_train)
        
        # Fit treatment model
        if verbose:
            print(f"Fitting treatment model (n={len(X_treatment_train):,})...")
        
        self.model_1 = self.treatment_estimator
        self.model_1.fit(X_treatment_train, y_treatment_train)
        
        # Evaluate on test set
        y_pred_0 = self.model_0.predict_proba(X_test)[:, 1]
        y_pred_1 = self.model_1.predict_proba(X_test)[:, 1]
        
        mask_control_test = T_test == 0
        mask_treatment_test = T_test == 1
        
        auc_0 = roc_auc_score(y_test[mask_control_test], y_pred_0[mask_control_test])
        auc_1 = roc_auc_score(y_test[mask_treatment_test], y_pred_1[mask_treatment_test])
        
        if verbose:
            print(f"\nControl model AUC: {auc_0:.4f}")
            print(f"Treatment model AUC: {auc_1:.4f}")
        
        self.is_fitted = True
        
        # Store metrics
        self._train_metrics = {
            'auc_0': auc_0,
            'auc_1': auc_1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_control': (T_train == 0).sum(),
            'n_treatment': (T_train == 1).sum()
        }
        
        return self._train_metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict CATE (uplift)
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Features
        
        Returns
        -------
        cate : array, shape (n_samples,)
            Conditional Average Treatment Effect (uplift scores)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        p_0 = self.model_0.predict_proba(X_scaled)[:, 1]
        p_1 = self.model_1.predict_proba(X_scaled)[:, 1]
        
        return p_1 - p_0
    
    def predict_cate(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Predict CATE with detailed outputs
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Features
        
        Returns
        -------
        predictions : dict
            Dictionary containing:
            - 'p_control': Control group probabilities
            - 'p_treatment': Treatment group probabilities
            - 'cate': CATE scores (p_treatment - p_control)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        p_0 = self.model_0.predict_proba(X_scaled)[:, 1]
        p_1 = self.model_1.predict_proba(X_scaled)[:, 1]
        
        return {
            'p_control': p_0,
            'p_treatment': p_1,
            'cate': p_1 - p_0
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to pickle file
        
        Parameters
        ----------
        filepath : str or Path
            Output file path
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_0': self.model_0,
            'model_1': self.model_1,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'train_metrics': self._train_metrics,
            'class': 'TLearner',
            'version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TLearner':
        """
        Load model from pickle file
        
        Parameters
        ----------
        filepath : str or Path
            Input file path
        
        Returns
        -------
        model : TLearner
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        model = cls(random_state=model_data.get('random_state', 42))
        
        # Restore state
        model.model_0 = model_data['model_0']
        model.model_1 = model_data['model_1']
        model.scaler = model_data['scaler']
        model.feature_cols = model_data['feature_cols']
        model.is_fitted = model_data['is_fitted']
        model._train_metrics = model_data.get('train_metrics', {})
        
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_cols) if self.feature_cols else None,
            'train_metrics': self._train_metrics
        }
    
    def __repr__(self) -> str:
        """String representation"""
        status = "fitted" if self.is_fitted else "not fitted"
        n_feat = len(self.feature_cols) if self.feature_cols else "unknown"
        return f"TLearner(random_state={self.random_state}, n_features={n_feat}, status={status})"