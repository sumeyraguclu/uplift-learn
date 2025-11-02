# ==================== src/__init__.py ====================
"""
Uplift Modeling Package

Production-grade uplift modeling components:
- model: TLearner and other uplift models
- metrics: Qini AUC, Uplift@k, and evaluation metrics
- optimize: Campaign optimization and ROI calculation
- config: Configuration management
- calibration: CATE calibration
"""

__version__ = '1.0.0'
__author__ = 'Sumeyra Guclu'

# Import key classes and functions for convenience
from .model import TLearner
from .metrics import (
    qini_auc_score,
    uplift_at_k,
    uplift_at_k_multiple,
    average_treatment_effect,
    treatment_balance_check,
    evaluate_uplift_model,
    qini_curve_data
)
from .optimize import (
    greedy_optimizer,
    roi_threshold_optimizer,
    top_k_optimizer,
    compare_strategies,
    calculate_campaign_metrics,
    optimize_with_constraints
)
from .config import (
    Config,
    get_config,
    load_config
)
from .calibration import (
    CATECalibrator,
    calibrate_cate
)

__all__ = [
    # Models
    'TLearner',
    
    # Metrics
    'qini_auc_score',
    'uplift_at_k',
    'uplift_at_k_multiple',
    'average_treatment_effect',
    'treatment_balance_check',
    'evaluate_uplift_model',
    'qini_curve_data',
    
    # Optimization
    'greedy_optimizer',
    'roi_threshold_optimizer',
    'top_k_optimizer',
    'compare_strategies',
    'calculate_campaign_metrics',
    'optimize_with_constraints',
    
    # Configuration
    'Config',
    'get_config',
    'load_config',
    
    # Calibration
    'CATECalibrator',
    'calibrate_cate',
]