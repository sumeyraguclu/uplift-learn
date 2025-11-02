# ==================== src/metrics.py ====================
"""
Uplift modeling metrikleri

Production-grade uplift evaluation metrics
Reference: scikit-uplift and Radcliffe (2007)
"""

import numpy as np
from typing import Union, List, Dict
from scipy import stats


def qini_auc_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray
) -> float:
    """
    Qini AUC (Area Under Qini Curve)
    
    Uplift modelinin performans metriği. Model müşterileri uplift'e göre 
    sıraladığında ne kadar iyi ayrıştırma yapıyor?
    
    Theory:
    -------
    Qini eğrisi, müşterileri tahmin edilen uplift'e göre sıraladığımızda
    kümülatif uplift kazancını gösterir:
    
    Qini(p) = N_t(p)/N_t * Y_t(p) - N_c(p)/N_c * Y_c(p)
    
    Where:
    - N_t(p): Top p%'deki treatment samples sayısı
    - Y_t(p): Top p%'deki treatment outcomes toplamı
    - N_c(p), Y_c(p): Control için aynısı
    
    Yüksek Qini AUC = Model yüksek uplift'li müşterileri başarıyla tanımlıyor
    
    Interpretation:
    - > 0.15: EXCELLENT - Model çok iyi ayrıştırıyor
    - 0.05-0.15: GOOD - Model makul derecede başarılı
    - 0.0-0.05: FAIR - Model zayıf ama sıfırdan iyi
    - < 0: POOR - Random'dan daha kötü
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Gerçek outcome (0 veya 1)
    uplift : array-like, shape (n_samples,)
        Model uplift tahminleri (CATE scores)
    treatment : array-like, shape (n_samples,)
        Treatment göstergesi (0 veya 1)
    
    Returns
    -------
    qini_auc : float
        Qini AUC skoru (normalized)
    
    Examples
    --------
    >>> from src.metrics import qini_auc_score
    >>> qini = qini_auc_score(y_test, uplift_pred, treatment_test)
    >>> print(f"Qini AUC: {qini:.4f}")
    >>> 
    >>> if qini > 0.15:
    ...     print("EXCELLENT model performance!")
    """
    y_true = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    
    # Sort by predicted uplift (descending)
    idx_sorted = np.argsort(uplift)[::-1]
    y_sorted = y_true[idx_sorted]
    t_sorted = treatment[idx_sorted]
    
    n = len(y_true)
    n_treatment = (t_sorted == 1).sum()
    n_control = (t_sorted == 0).sum()
    
    # Edge case: no treatment or control samples
    if n_treatment == 0 or n_control == 0:
        return 0.0
    
    # Separate by group
    y_treatment = y_sorted[t_sorted == 1]
    y_control = y_sorted[t_sorted == 0]
    
    # Cumulative sums
    cum_treatment = np.cumsum(y_treatment)
    cum_control = np.cumsum(y_control)
    
    # Cumulative response rates
    rate_treatment = cum_treatment / np.arange(1, len(y_treatment) + 1)
    rate_control = cum_control / np.arange(1, len(y_control) + 1)
    
    # Qini calculation: integral of difference in response rates
    # Normalized by total sample size
    qini = np.trapz(rate_treatment) / n - np.trapz(rate_control) / n
    
    # Clip to reasonable bounds
    return np.clip(qini, -1, 1)


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.3
) -> float:
    """
    Uplift@k: İlk k% müşteride gerçekleşen ortalama uplift
    
    Business interpretation: En yüksek uplift skorlu %k müşteriye kampanya 
    yaparsak, beklenen ortalama uplift ne olur?
    
    Theory:
    -------
    1. Müşterileri tahmin edilen uplift'e göre sırala (azalan)
    2. İlk k% müşteriyi seç
    3. Bu gruptaki treatment vs control conversion farkını hesapla
    
    Uplift@k = E[Y|T=1, top k%] - E[Y|T=0, top k%]
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Gerçek outcome (0 veya 1)
    uplift : array-like, shape (n_samples,)
        Model uplift tahminleri
    treatment : array-like, shape (n_samples,)
        Treatment göstergesi (0 veya 1)
    k : float, optional
        Hedeflenen yüzde (0-1 arası). Default: 0.3 (top 30%)
    
    Returns
    -------
    uplift_k : float
        İlk k%'deki gerçekleşen ortalama uplift (0-1 scale)
    
    Examples
    --------
    >>> from src.metrics import uplift_at_k
    >>> 
    >>> # Top 30% için uplift
    >>> u30 = uplift_at_k(y_test, uplift_pred, treatment_test, k=0.3)
    >>> print(f"Uplift@30%: {u30*100:.2f}%")
    >>> 
    >>> # Farklı k değerleri için karşılaştırma
    >>> for k_val in [0.1, 0.2, 0.3, 0.5]:
    ...     u = uplift_at_k(y_test, uplift_pred, treatment_test, k=k_val)
    ...     print(f"Top {k_val*100:.0f}%: {u*100:+.2f}%")
    """
    y_true = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}")
    
    # Sort by predicted uplift (descending)
    idx_sorted = np.argsort(uplift)[::-1]
    
    # Select top k%
    n_k = int(len(y_true) * k)
    if n_k == 0:
        n_k = 1  # At least 1 sample
    
    y_topk = y_true[idx_sorted[:n_k]]
    t_topk = treatment[idx_sorted[:n_k]]
    
    # Calculate uplift in top k%
    mask_treatment = t_topk == 1
    mask_control = t_topk == 0
    
    # Need both groups in top k%
    if mask_treatment.sum() == 0 or mask_control.sum() == 0:
        return 0.0
    
    uplift_k = y_topk[mask_treatment].mean() - y_topk[mask_control].mean()
    
    return uplift_k


def uplift_at_k_multiple(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k_list: List[float] = [0.1, 0.2, 0.3, 0.5]
) -> Dict[str, float]:
    """
    Multiple k değerleri için Uplift@k hesapla
    
    Parameters
    ----------
    y_true : array-like
        Gerçek outcome
    uplift : array-like
        Model uplift tahminleri
    treatment : array-like
        Treatment göstergesi
    k_list : list of float
        k değerleri listesi (0-1 arası)
    
    Returns
    -------
    results : dict
        Her k için uplift değerleri
        Format: {'uplift_at_10': 0.05, 'uplift_at_20': 0.04, ...}
    
    Examples
    --------
    >>> results = uplift_at_k_multiple(y, uplift, treatment, k_list=[0.1, 0.3, 0.5])
    >>> for k, v in results.items():
    ...     print(f"{k}: {v*100:+.2f}%")
    """
    results = {}
    
    for k in k_list:
        u = uplift_at_k(y_true, uplift, treatment, k=k)
        # Convert to percentage notation
        k_pct = int(k * 100)
        results[f'uplift_at_{k_pct}'] = u * 100  # Store as percentage
    
    return results


def average_treatment_effect(
    y_true: np.ndarray,
    treatment: np.ndarray,
    conf: float = 0.95
) -> Dict[str, float]:
    """
    Average Treatment Effect (ATE) with confidence interval
    
    ATE = E[Y|T=1] - E[Y|T=0]
    
    Overall average effect of treatment across entire population
    
    Parameters
    ----------
    y_true : array-like
        Gerçek outcome (0 veya 1)
    treatment : array-like
        Treatment göstergesi (0 veya 1)
    conf : float, optional
        Confidence level. Default: 0.95 (95% CI)
    
    Returns
    -------
    results : dict
        'ate': Average treatment effect
        'ci_lower': Confidence interval lower bound
        'ci_upper': Confidence interval upper bound
        'se': Standard error
    
    Examples
    --------
    >>> from src.metrics import average_treatment_effect
    >>> ate_results = average_treatment_effect(y_true, treatment)
    >>> print(f"ATE: {ate_results['ate']*100:.2f}%")
    >>> print(f"95% CI: [{ate_results['ci_lower']*100:.2f}%, {ate_results['ci_upper']*100:.2f}%]")
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    
    y_treatment = y_true[treatment == 1]
    y_control = y_true[treatment == 0]
    
    # ATE
    ate = y_treatment.mean() - y_control.mean()
    
    # Standard error
    se_treatment = y_treatment.std() / np.sqrt(len(y_treatment))
    se_control = y_control.std() / np.sqrt(len(y_control))
    se = np.sqrt(se_treatment**2 + se_control**2)
    
    # Confidence interval
    z = stats.norm.ppf((1 + conf) / 2)
    ci_lower = ate - z * se
    ci_upper = ate + z * se
    
    return {
        'ate': ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se
    }


def treatment_balance_check(
    X: np.ndarray,
    treatment: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Union[float, str, bool]]:
    """
    Treatment balance check using Standardized Mean Difference (SMD)
    
    Checks whether treatment and control groups are balanced on covariates.
    Important for causal inference validity.
    
    SMD = |mean(X_t) - mean(X_c)| / sqrt((var(X_t) + var(X_c)) / 2)
    
    Interpretation:
    - SMD < 0.1: Good balance
    - 0.1 ≤ SMD < 0.2: Acceptable balance
    - SMD ≥ 0.2: Poor balance (potential confounding)
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Covariate matrix
    treatment : array-like, shape (n_samples,)
        Treatment indicator
    threshold : float, optional
        SMD threshold for "good" balance. Default: 0.1
    
    Returns
    -------
    results : dict
        'avg_smd': Average SMD across features
        'max_smd': Maximum SMD across features
        'is_balanced': Whether avg SMD < threshold
        'status': 'Good', 'OK', or 'Poor'
    
    Examples
    --------
    >>> from src.metrics import treatment_balance_check
    >>> balance = treatment_balance_check(X_test, treatment_test)
    >>> print(f"Balance: {balance['status']} (SMD={balance['avg_smd']:.4f})")
    """
    X = np.asarray(X)
    treatment = np.asarray(treatment)
    
    X_treatment = X[treatment == 1]
    X_control = X[treatment == 0]
    
    n_features = X.shape[1]
    smd_list = []
    
    for i in range(n_features):
        x_t = X_treatment[:, i]
        x_c = X_control[:, i]
        
        # Skip if no variance
        if x_t.std() + x_c.std() == 0:
            continue
        
        # Standardized mean difference
        smd = abs(x_t.mean() - x_c.mean()) / np.sqrt((x_t.var() + x_c.var()) / 2 + 1e-10)
        smd_list.append(smd)
    
    if not smd_list:
        avg_smd = 0.0
        max_smd = 0.0
    else:
        avg_smd = np.mean(smd_list)
        max_smd = np.max(smd_list)
    
    # Determine status
    if avg_smd < 0.1:
        status = 'Good'
    elif avg_smd < 0.2:
        status = 'OK'
    else:
        status = 'Poor'
    
    return {
        'avg_smd': avg_smd,
        'max_smd': max_smd,
        'is_balanced': avg_smd < threshold,
        'status': status
    }


def qini_curve_data(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Generate Qini curve data for plotting
    
    Returns x and y coordinates for Qini curve visualization
    
    Parameters
    ----------
    y_true : array-like
        Gerçek outcome
    uplift : array-like
        Model uplift tahminleri
    treatment : array-like
        Treatment göstergesi
    n_points : int, optional
        Number of points in curve. Default: 100
    
    Returns
    -------
    curve_data : dict
        'x': Percentage of population targeted (0-100)
        'y': Cumulative uplift at each percentage
    
    Examples
    --------
    >>> from src.metrics import qini_curve_data
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> curve = qini_curve_data(y_true, uplift, treatment)
    >>> plt.plot(curve['x'], curve['y'])
    >>> plt.xlabel('Population Targeted (%)')
    >>> plt.ylabel('Cumulative Uplift (%)')
    >>> plt.show()
    """
    y_true = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    
    # Sort by predicted uplift
    idx = np.argsort(uplift)[::-1]
    y_sorted = y_true[idx]
    t_sorted = treatment[idx]
    
    n = len(y_true)
    step = max(1, n // n_points)
    
    x_vals = []
    y_vals = []
    
    for i in range(step, n + 1, step):
        # Current subset
        y_sub = y_sorted[:i]
        t_sub = t_sorted[:i]
        
        mask_treatment = t_sub == 1
        mask_control = t_sub == 0
        
        # Calculate uplift if both groups present
        if mask_treatment.sum() > 0 and mask_control.sum() > 0:
            uplift_val = (y_sub[mask_treatment].mean() - y_sub[mask_control].mean()) * 100
            x_vals.append((i / n) * 100)
            y_vals.append(uplift_val)
    
    return {
        'x': np.array(x_vals),
        'y': np.array(y_vals)
    }


def evaluate_uplift_model(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray = None,
    k_list: List[float] = [0.1, 0.2, 0.3, 0.5]
) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive uplift model evaluation
    
    One-stop function for all common uplift metrics
    
    Parameters
    ----------
    y_true : array-like
        Gerçek outcome
    uplift : array-like
        Model uplift predictions
    treatment : array-like
        Treatment indicator
    X : array-like, optional
        Covariates for balance check
    k_list : list of float, optional
        k values for uplift@k
    
    Returns
    -------
    metrics : dict
        Complete evaluation results
    
    Examples
    --------
    >>> from src.metrics import evaluate_uplift_model
    >>> 
    >>> metrics = evaluate_uplift_model(
    ...     y_true, uplift_pred, treatment, 
    ...     X=X_test, 
    ...     k_list=[0.1, 0.2, 0.3]
    ... )
    >>> 
    >>> print(f"Qini AUC: {metrics['qini_auc']:.4f}")
    >>> print(f"ATE: {metrics['ate']['ate']*100:.2f}%")
    >>> for k, v in metrics['uplift_at_k'].items():
    ...     print(f"{k}: {v:+.2f}%")
    """
    results = {
        'qini_auc': qini_auc_score(y_true, uplift, treatment),
        'uplift_at_k': uplift_at_k_multiple(y_true, uplift, treatment, k_list),
        'ate': average_treatment_effect(y_true, treatment)
    }
    
    # Add balance check if covariates provided
    if X is not None:
        results['balance'] = treatment_balance_check(X, treatment)
    
    return results