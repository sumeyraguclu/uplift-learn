"""
Calibration Module Usage Examples
"""

import numpy as np
import pandas as pd
from src.calibration import CATECalibrator, calibrate_cate


# ======================== EXAMPLE 1: Basic Calibration ========================

def example_basic():
    """Basic CATE calibration"""
    print("=" * 70)
    print("EXAMPLE 1: Basic CATE Calibration")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 2000
    
    # Features
    X = np.random.randn(n, 5)
    
    # Treatment assignment
    treatment = np.random.binomial(1, 0.5, n)
    
    # True probabilities (with treatment effect)
    true_p_control = 1 / (1 + np.exp(-X[:, 0]))
    true_uplift = X[:, 0] * 0.15 + X[:, 1] * 0.05
    true_p_treatment = np.clip(true_p_control + true_uplift, 0, 1)
    
    # Generate outcomes
    y = np.where(
        treatment == 1,
        np.random.binomial(1, true_p_treatment),
        np.random.binomial(1, true_p_control)
    )
    
    # Biased predictions (overly optimistic)
    pred_p_treatment = np.clip(true_p_treatment + 0.1, 0, 1)
    pred_p_control = np.clip(true_p_control + 0.05, 0, 1)
    
    print(f"\nData generated:")
    print(f"  Samples: {n}")
    print(f"  Treatment: {(treatment==1).sum()} | Control: {(treatment==0).sum()}")
    print(f"  True mean uplift: {true_uplift.mean():.4f}")
    print(f"  Predicted mean uplift: {(pred_p_treatment - pred_p_control).mean():.4f}")
    
    # Calibrate
    print("\nCalibrating...")
    calibrator = CATECalibrator()
    calibrated = calibrator.fit_transform(
        pred_p_treatment, pred_p_control, y, treatment
    )
    
    print(f"\nResults:")
    print(f"  Raw CATE: {(pred_p_treatment - pred_p_control).mean():.4f}")
    print(f"  Calibrated CATE: {calibrated['cate'].mean():.4f}")
    print(f"  True CATE: {true_uplift.mean():.4f}")
    print(f"  Improvement: Closer to true value!")


# ======================== EXAMPLE 2: Save and Load ========================

def example_save_load():
    """Save and load calibrator"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Save and Load Calibrator")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 1000
    
    p_treatment = np.random.beta(3, 2, n)
    p_control = np.random.beta(2, 3, n)
    treatment = np.random.binomial(1, 0.5, n)
    
    # Generate outcomes with bias
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment * 0.9),  # Model overestimates
        np.random.binomial(1, p_control * 0.9)
    )
    
    # Fit and save
    print("\nFitting calibrator...")
    calibrator = CATECalibrator()
    calibrator.fit(p_treatment, p_control, y, treatment, verbose=False)
    
    calibrator.save('models/example_calibrator.pkl')
    print("Saved: models/example_calibrator.pkl")
    
    # Load and use
    print("\nLoading calibrator...")
    loaded_calibrator = CATECalibrator.load('models/example_calibrator.pkl')
    
    # Apply to new data
    new_p_treatment = np.random.beta(3, 2, 100)
    new_p_control = np.random.beta(2, 3, 100)
    
    calibrated = loaded_calibrator.transform(new_p_treatment, new_p_control)
    
    print(f"Applied to new data:")
    print(f"  Raw CATE mean: {(new_p_treatment - new_p_control).mean():.4f}")
    print(f"  Calibrated CATE mean: {calibrated['cate'].mean():.4f}")


# ======================== EXAMPLE 3: Convenience Function ========================

def example_convenience():
    """Using calibrate_cate convenience function"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Convenience Function")
    print("=" * 70)
    
    # Create sample DataFrames
    np.random.seed(42)
    n = 1500
    
    # Predictions DataFrame
    predictions_df = pd.DataFrame({
        'client_id': range(n),
        'p_treatment': np.random.beta(3, 2, n),
        'p_control': np.random.beta(2, 3, n)
    })
    predictions_df['cate'] = predictions_df['p_treatment'] - predictions_df['p_control']
    
    # Outcomes DataFrame
    treatment = np.random.binomial(1, 0.5, n)
    y = np.where(
        treatment == 1,
        np.random.binomial(1, predictions_df['p_treatment'] * 0.95),
        np.random.binomial(1, predictions_df['p_control'] * 0.95)
    )
    
    outcomes_df = pd.DataFrame({
        'client_id': range(n),
        'treatment': treatment,
        'target': y
    })
    
    print(f"\nInput data:")
    print(f"  Predictions: {len(predictions_df)} rows")
    print(f"  Outcomes: {len(outcomes_df)} rows")
    
    # Calibrate
    calibrated_df, calibrator = calibrate_cate(
        predictions_df=predictions_df,
        outcomes_df=outcomes_df,
        save_calibrator=False,
        verbose=True
    )
    
    print(f"\nOutput:")
    print(f"  Calibrated DataFrame: {len(calibrated_df)} rows")
    print(f"  New columns: {[c for c in calibrated_df.columns if 'cal' in c]}")


# ======================== EXAMPLE 4: Calibration Metrics ========================

def example_metrics():
    """Examining calibration metrics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Calibration Metrics")
    print("=" * 70)
    
    # Generate data with strong bias
    np.random.seed(42)
    n = 2000
    
    treatment = np.random.binomial(1, 0.5, n)
    
    # True probabilities
    true_p = 0.3 + np.random.randn(n) * 0.1
    true_p = np.clip(true_p, 0, 1)
    
    # Biased predictions (overconfident)
    pred_p_treatment = np.clip(true_p + 0.15, 0, 1)
    pred_p_control = np.clip(true_p + 0.10, 0, 1)
    
    # Generate outcomes
    y = np.random.binomial(1, true_p)
    
    # Calibrate
    calibrator = CATECalibrator()
    calibrator.fit(pred_p_treatment, pred_p_control, y, treatment, verbose=False)
    
    # Get metrics
    metrics = calibrator.get_metrics()
    
    print("\nCalibration Metrics:")
    print(f"  Treatment Group:")
    print(f"    MAE before: {metrics['mae_before_treatment']:.4f}")
    print(f"    MAE after: {metrics['mae_after_treatment']:.4f}")
    print(f"    Improvement: {metrics['improvement_treatment']*100:.1f}%")
    
    print(f"\n  Control Group:")
    print(f"    MAE before: {metrics['mae_before_control']:.4f}")
    print(f"    MAE after: {metrics['mae_after_control']:.4f}")
    print(f"    Improvement: {metrics['improvement_control']*100:.1f}%")


# ======================== EXAMPLE 5: Calibration Curves ========================

def example_plots():
    """Generate calibration curve plots"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Calibration Curves")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 3000
    
    treatment = np.random.binomial(1, 0.5, n)
    
    # Predictions with systematic bias
    p_treatment_raw = np.random.beta(4, 2, n)
    p_control_raw = np.random.beta(2, 4, n)
    
    # Add overconfidence bias
    p_treatment = np.clip(p_treatment_raw * 1.2, 0, 1)
    p_control = np.clip(p_control_raw * 0.8, 0, 1)
    
    # Generate outcomes
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment_raw),
        np.random.binomial(1, p_control_raw)
    )
    
    print(f"\nGenerating calibration plot...")
    
    # Fit calibrator
    calibrator = CATECalibrator()
    calibrator.fit(p_treatment, p_control, y, treatment, verbose=False)
    
    # Plot
    calibrator.plot_calibration(
        p_treatment, p_control, y, treatment,
        save_path='plots/example_calibration_curves.png'
    )
    
    print("Saved: plots/example_calibration_curves.png")


# ======================== EXAMPLE 6: Impact on ROI ========================

def example_roi_impact():
    """Show calibration impact on ROI calculation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Calibration Impact on ROI")
    print("=" * 70)
    
    from src.optimize import greedy_optimizer
    
    # Generate data
    np.random.seed(42)
    n = 5000
    
    treatment = np.random.binomial(1, 0.5, n)
    
    # True uplift
    true_uplift = np.random.beta(2, 8, n) * 0.15
    
    # Biased predictions (overestimate)
    biased_uplift = true_uplift * 1.5
    
    # Generate outcomes
    base_p = 0.2
    y = np.where(
        treatment == 1,
        np.random.binomial(1, base_p + true_uplift),
        np.random.binomial(1, base_p)
    )
    
    # Calibrate
    p_treatment_biased = base_p + biased_uplift
    p_control_biased = np.full(n, base_p)
    
    calibrator = CATECalibrator()
    calibrated = calibrator.fit_transform(
        p_treatment_biased, p_control_biased, y, treatment, verbose=False
    )
    
    # Compare ROI calculations
    margin = 50.0
    cost = 0.5
    budget = 2000.0
    
    print("\nUsing BIASED predictions:")
    result_biased = greedy_optimizer(biased_uplift, margin, cost, budget)
    print(f"  Expected ROI: {result_biased['roi_pct']:.1f}%")
    
    print("\nUsing CALIBRATED predictions:")
    result_calibrated = greedy_optimizer(calibrated['cate'], margin, cost, budget)
    print(f"  Expected ROI: {result_calibrated['roi_pct']:.1f}%")
    
    print("\nUsing TRUE uplift (benchmark):")
    result_true = greedy_optimizer(true_uplift, margin, cost, budget)
    print(f"  Expected ROI: {result_true['roi_pct']:.1f}%")
    
    print(f"\nCalibration brings ROI estimate closer to true value!")
    print(f"  Biased error: {abs(result_biased['roi_pct'] - result_true['roi_pct']):.1f}pp")
    print(f"  Calibrated error: {abs(result_calibrated['roi_pct'] - result_true['roi_pct']):.1f}pp")


# ======================== RUN ALL EXAMPLES ========================

if __name__ == '__main__':
    example_basic()
    example_save_load()
    example_convenience()
    example_metrics()
    example_plots()
    example_roi_impact()
    
    print("\n" + "=" * 70)
    print("ALL CALIBRATION EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Calibration reduces prediction bias")
    print("  • Improves ROI estimate accuracy")
    print("  • CATECalibrator is reusable (save/load)")
    print("  • calibrate_cate() is convenient for DataFrames")
    print("  • Always check calibration metrics")