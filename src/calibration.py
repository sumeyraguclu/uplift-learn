# ==================== src/calibration.py ====================
"""
CATE Calibration Module

Calibrates raw CATE predictions to improve reliability of
uplift estimates for financial planning.

Theory:
-------
Raw model predictions may be biased. Calibration adjusts predictions
to match actual observed uplift, improving ROI estimates.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from sklearn.calibration import IsotonicRegression, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class CATECalibrator:
    """
    CATE Calibration using Isotonic Regression
    
    Calibrates raw CATE predictions to match actual observed uplift.
    Uses separate calibration for treatment and control probabilities.
    
    Theory:
    -------
    1. Split data into calibration and validation sets
    2. Fit isotonic regression: observed ~ predicted for each group
    3. Apply calibration to new predictions
    4. Calibrated CATE = calibrated_p_treatment - calibrated_p_control
    
    Parameters
    ----------
    method : str, optional
        Calibration method. Options: 'isotonic', 'sigmoid'
        Default: 'isotonic'
    cv_folds : int, optional
        Cross-validation folds. Default: 5
    
    Attributes
    ----------
    calibrator_treatment : IsotonicRegression
        Fitted treatment calibrator
    calibrator_control : IsotonicRegression
        Fitted control calibrator
    is_fitted : bool
        Whether calibrator has been fitted
    
    Examples
    --------
    >>> from src.calibration import CATECalibrator
    >>> 
    >>> calibrator = CATECalibrator()
    >>> calibrator.fit(p_treatment, p_control, y_true, treatment)
    >>> 
    >>> # Apply to new data
    >>> calibrated = calibrator.transform(p_treatment_new, p_control_new)
    >>> cate_calibrated = calibrated['cate']
    """
    
    def __init__(self, method: str = 'isotonic', cv_folds: int = 5):
        self.method = method
        self.cv_folds = cv_folds
        self.calibrator_treatment = None
        self.calibrator_control = None
        self.is_fitted = False
        self._fit_metrics = {}
    
    def fit(
        self,
        p_treatment: np.ndarray,
        p_control: np.ndarray,
        y_true: np.ndarray,
        treatment: np.ndarray,
        verbose: bool = True
    ) -> 'CATECalibrator':
        """
        Fit calibrators on actual outcomes
        
        Parameters
        ----------
        p_treatment : array-like
            Raw treatment probability predictions
        p_control : array-like
            Raw control probability predictions
        y_true : array-like
            Actual outcomes (0 or 1)
        treatment : array-like
            Treatment assignment (0 or 1)
        verbose : bool, optional
            Print fitting progress. Default: True
        
        Returns
        -------
        self : CATECalibrator
            Fitted calibrator
        """
        p_treatment = np.asarray(p_treatment)
        p_control = np.asarray(p_control)
        y_true = np.asarray(y_true)
        treatment = np.asarray(treatment)
        
        if verbose:
            print("Fitting CATE Calibrator...")
            print(f"  Method: {self.method}")
            print(f"  Samples: {len(y_true):,}")
        
        # Separate by treatment group
        mask_treatment = treatment == 1
        mask_control = treatment == 0
        
        # Treatment group calibration
        if verbose:
            print(f"\n  Treatment group: {mask_treatment.sum():,} samples")
        
        self.calibrator_treatment = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_treatment.fit(
            p_treatment[mask_treatment],
            y_true[mask_treatment]
        )
        
        # Control group calibration
        if verbose:
            print(f"  Control group: {mask_control.sum():,} samples")
        
        self.calibrator_control = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_control.fit(
            p_control[mask_control],
            y_true[mask_control]
        )
        
        # Calculate calibration metrics
        cal_p_treatment = self.calibrator_treatment.transform(p_treatment[mask_treatment])
        cal_p_control = self.calibrator_control.transform(p_control[mask_control])
        
        # Before calibration
        mae_before_t = np.abs(p_treatment[mask_treatment] - y_true[mask_treatment]).mean()
        mae_before_c = np.abs(p_control[mask_control] - y_true[mask_control]).mean()
        
        # After calibration
        mae_after_t = np.abs(cal_p_treatment - y_true[mask_treatment]).mean()
        mae_after_c = np.abs(cal_p_control - y_true[mask_control]).mean()
        
        self._fit_metrics = {
            'mae_before_treatment': mae_before_t,
            'mae_before_control': mae_before_c,
            'mae_after_treatment': mae_after_t,
            'mae_after_control': mae_after_c,
            'improvement_treatment': (mae_before_t - mae_after_t) / mae_before_t,
            'improvement_control': (mae_before_c - mae_after_c) / mae_before_c
        }
        
        if verbose:
            print(f"\n  Treatment MAE: {mae_before_t:.4f} → {mae_after_t:.4f}")
            print(f"  Control MAE: {mae_before_c:.4f} → {mae_after_c:.4f}")
            print(f"  Improvement: {self._fit_metrics['improvement_treatment']*100:.1f}% / "
                  f"{self._fit_metrics['improvement_control']*100:.1f}%")
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        p_treatment: np.ndarray,
        p_control: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Apply calibration to predictions
        
        Parameters
        ----------
        p_treatment : array-like
            Raw treatment probability predictions
        p_control : array-like
            Raw control probability predictions
        
        Returns
        -------
        result : dict
            Dictionary with:
            - 'p_treatment_cal': Calibrated treatment probabilities
            - 'p_control_cal': Calibrated control probabilities
            - 'cate': Calibrated CATE (difference)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        p_treatment = np.asarray(p_treatment)
        p_control = np.asarray(p_control)
        
        # Apply calibration
        p_treatment_cal = self.calibrator_treatment.transform(p_treatment)
        p_control_cal = self.calibrator_control.transform(p_control)
        
        # Calculate calibrated CATE
        cate_cal = p_treatment_cal - p_control_cal
        
        return {
            'p_treatment_cal': p_treatment_cal,
            'p_control_cal': p_control_cal,
            'cate': cate_cal
        }
    
    def fit_transform(
        self,
        p_treatment: np.ndarray,
        p_control: np.ndarray,
        y_true: np.ndarray,
        treatment: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Fit calibrator and transform in one step
        
        Parameters
        ----------
        p_treatment : array-like
            Raw treatment predictions
        p_control : array-like
            Raw control predictions
        y_true : array-like
            Actual outcomes
        treatment : array-like
            Treatment assignment
        verbose : bool, optional
            Print progress. Default: True
        
        Returns
        -------
        result : dict
            Calibrated predictions
        """
        self.fit(p_treatment, p_control, y_true, treatment, verbose)
        return self.transform(p_treatment, p_control)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save calibrator to file"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'calibrator_treatment': self.calibrator_treatment,
            'calibrator_control': self.calibrator_control,
            'method': self.method,
            'cv_folds': self.cv_folds,
            'is_fitted': self.is_fitted,
            'fit_metrics': self._fit_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CATECalibrator':
        """Load calibrator from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data['method'], cv_folds=data['cv_folds'])
        calibrator.calibrator_treatment = data['calibrator_treatment']
        calibrator.calibrator_control = data['calibrator_control']
        calibrator.is_fitted = data['is_fitted']
        calibrator._fit_metrics = data.get('fit_metrics', {})
        
        return calibrator
    
    def get_metrics(self) -> Dict:
        """Get calibration fit metrics"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        return self._fit_metrics.copy()
    
    def plot_calibration(
        self,
        p_treatment: np.ndarray,
        p_control: np.ndarray,
        y_true: np.ndarray,
        treatment: np.ndarray,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot calibration curves
        
        Parameters
        ----------
        p_treatment : array-like
            Raw treatment predictions
        p_control : array-like
            Raw control predictions
        y_true : array-like
            Actual outcomes
        treatment : array-like
            Treatment assignment
        save_path : str or Path, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Treatment group
        mask_t = treatment == 1
        p_t = p_treatment[mask_t]
        y_t = y_true[mask_t]
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_means_t = []
        bin_observed_t = []
        for i in range(len(bins)-1):
            mask = (p_t >= bins[i]) & (p_t < bins[i+1])
            if mask.sum() > 0:
                bin_means_t.append(p_t[mask].mean())
                bin_observed_t.append(y_t[mask].mean())
        
        ax1.scatter(bin_means_t, bin_observed_t, s=100, alpha=0.6, label='Actual')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Calibrated
        p_t_cal = self.calibrator_treatment.transform(p_t)
        bin_means_t_cal = []
        bin_observed_t_cal = []
        for i in range(len(bins)-1):
            mask = (p_t_cal >= bins[i]) & (p_t_cal < bins[i+1])
            if mask.sum() > 0:
                bin_means_t_cal.append(p_t_cal[mask].mean())
                bin_observed_t_cal.append(y_t[mask].mean())
        
        ax1.scatter(bin_means_t_cal, bin_observed_t_cal, s=100, alpha=0.6, 
                   marker='s', label='Calibrated')
        ax1.set_xlabel('Predicted Probability', fontsize=11)
        ax1.set_ylabel('Observed Frequency', fontsize=11)
        ax1.set_title('Treatment Group Calibration', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Control group
        mask_c = treatment == 0
        p_c = p_control[mask_c]
        y_c = y_true[mask_c]
        
        bin_means_c = []
        bin_observed_c = []
        for i in range(len(bins)-1):
            mask = (p_c >= bins[i]) & (p_c < bins[i+1])
            if mask.sum() > 0:
                bin_means_c.append(p_c[mask].mean())
                bin_observed_c.append(y_c[mask].mean())
        
        ax2.scatter(bin_means_c, bin_observed_c, s=100, alpha=0.6, label='Actual')
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Calibrated
        p_c_cal = self.calibrator_control.transform(p_c)
        bin_means_c_cal = []
        bin_observed_c_cal = []
        for i in range(len(bins)-1):
            mask = (p_c_cal >= bins[i]) & (p_c_cal < bins[i+1])
            if mask.sum() > 0:
                bin_means_c_cal.append(p_c_cal[mask].mean())
                bin_observed_c_cal.append(y_c[mask].mean())
        
        ax2.scatter(bin_means_c_cal, bin_observed_c_cal, s=100, alpha=0.6,
                   marker='s', label='Calibrated')
        ax2.set_xlabel('Predicted Probability', fontsize=11)
        ax2.set_ylabel('Observed Frequency', fontsize=11)
        ax2.set_title('Control Group Calibration', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def calibrate_cate(
    predictions_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    client_id_col: str = 'client_id',
    treatment_col: str = 'treatment',
    target_col: str = 'target',
    p_treatment_col: str = 'p_treatment',
    p_control_col: str = 'p_control',
    save_calibrator: bool = True,
    calibrator_path: str = 'models/calibrator.pkl',
    verbose: bool = True
) -> Tuple[pd.DataFrame, CATECalibrator]:
    """
    Convenience function to calibrate CATE predictions
    
    Parameters
    ----------
    predictions_df : DataFrame
        DataFrame with model predictions
    outcomes_df : DataFrame
        DataFrame with actual outcomes
    client_id_col : str
        Client ID column name
    treatment_col : str
        Treatment column name
    target_col : str
        Target column name
    p_treatment_col : str
        Treatment probability column
    p_control_col : str
        Control probability column
    save_calibrator : bool
        Whether to save calibrator to file
    calibrator_path : str
        Path to save calibrator
    verbose : bool
        Print progress
    
    Returns
    -------
    calibrated_df : DataFrame
        DataFrame with calibrated predictions
    calibrator : CATECalibrator
        Fitted calibrator object
    
    Examples
    --------
    >>> pred_df = pd.read_csv('results/tlearner_predictions.csv')
    >>> outcomes_df = pd.read_csv('data/outcomes.csv')
    >>> 
    >>> calibrated_df, calibrator = calibrate_cate(pred_df, outcomes_df)
    >>> calibrated_df.to_csv('results/calibrated_cate.csv', index=False)
    """
    # Merge predictions with outcomes
    merged = predictions_df.merge(
        outcomes_df[[client_id_col, treatment_col, target_col]],
        on=client_id_col
    )
    
    if verbose:
        print(f"Calibrating CATE predictions...")
        print(f"  Total samples: {len(merged):,}")
    
    # Extract arrays
    p_treatment = merged[p_treatment_col].values
    p_control = merged[p_control_col].values
    y_true = merged[target_col].values
    treatment = merged[treatment_col].values
    
    # Fit calibrator
    calibrator = CATECalibrator()
    calibrated = calibrator.fit_transform(
        p_treatment, p_control, y_true, treatment, verbose=verbose
    )
    
    # Create output DataFrame
    calibrated_df = predictions_df.copy()
    calibrated_df['p_treatment_cal'] = calibrator.transform(
        predictions_df[p_treatment_col].values,
        predictions_df[p_control_col].values
    )['p_treatment_cal']
    calibrated_df['p_control_cal'] = calibrator.transform(
        predictions_df[p_treatment_col].values,
        predictions_df[p_control_col].values
    )['p_control_cal']
    calibrated_df['cate_calibrated'] = calibrated_df['p_treatment_cal'] - calibrated_df['p_control_cal']
    
    if verbose:
        print(f"\nCalibration complete!")
        print(f"  Raw CATE mean: {predictions_df.get('cate', predictions_df[p_treatment_col] - predictions_df[p_control_col]).mean():+.4f}")
        print(f"  Calibrated CATE mean: {calibrated_df['cate_calibrated'].mean():+.4f}")
    
    # Save calibrator
    if save_calibrator:
        calibrator.save(calibrator_path)
        if verbose:
            print(f"  Saved calibrator: {calibrator_path}")
    
    return calibrated_df, calibrator