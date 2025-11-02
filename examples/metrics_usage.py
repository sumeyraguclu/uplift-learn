"""
Metrics Module Usage Examples
Demonstrates how to use src.metrics functions
"""

import numpy as np
import matplotlib.pyplot as plt
from src.metrics import (
    qini_auc_score,
    uplift_at_k,
    uplift_at_k_multiple,
    average_treatment_effect,
    treatment_balance_check,
    qini_curve_data,
    evaluate_uplift_model
)


# ======================== EXAMPLE 1: Basic Metrics ========================

def example_basic_metrics():
    """Basic metric calculations"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Metrics")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    
    # Features
    X = np.random.randn(n, 5)
    
    # Treatment assignment
    treatment = np.random.binomial(1, 0.5, n)
    
    # True uplift effect (heterogeneous)
    true_uplift = X[:, 0] * 0.15 + X[:, 1] * 0.05
    
    # Generate outcomes
    p_control = 1 / (1 + np.exp(-X[:, 0]))
    p_treatment = np.clip(p_control + true_uplift, 0, 1)
    
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment),
        np.random.binomial(1, p_control)
    )
    
    # Predicted CATE (with some noise)
    predicted_cate = true_uplift + np.random.randn(n) * 0.05
    
    # Calculate metrics
    print("\n[1] Qini AUC")
    qini = qini_auc_score(y, predicted_cate, treatment)
    print(f"Score: {qini:.4f}")
    
    print("\n[2] Uplift@k")
    u30 = uplift_at_k(y, predicted_cate, treatment, k=0.3)
    print(f"Uplift@30%: {u30*100:.2f}%")
    
    print("\n[3] Average Treatment Effect")
    ate = average_treatment_effect(y, treatment)
    print(f"ATE: {ate['ate']*100:.2f}%")
    print(f"95% CI: [{ate['ci_lower']*100:.2f}%, {ate['ci_upper']*100:.2f}%]")
    
    print("\n[4] Treatment Balance")
    balance = treatment_balance_check(X, treatment)
    print(f"Average SMD: {balance['avg_smd']:.4f}")
    print(f"Status: {balance['status']}")


# ======================== EXAMPLE 2: Multiple K Values ========================

def example_multiple_k():
    """Uplift@k for multiple thresholds"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multiple K Values")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n = 2000
    
    X = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)
    
    # Strong uplift for top segment
    uplift_scores = X[:, 0] * 0.2 + X[:, 1] * 0.1
    
    p_base = 0.3
    y = np.where(
        treatment == 1,
        np.random.binomial(1, np.clip(p_base + uplift_scores, 0, 1)),
        np.random.binomial(1, p_base)
    )
    
    # Multiple k values
    results = uplift_at_k_multiple(
        y, uplift_scores, treatment, 
        k_list=[0.05, 0.1, 0.2, 0.3, 0.5]
    )
    
    print("\nUplift at different thresholds:")
    for k, v in sorted(results.items()):
        k_pct = k.split('_')[-1]
        print(f"  Top {k_pct}%: {v:+.2f}%")


# ======================== EXAMPLE 3: Qini Curve Visualization ========================

def example_qini_curve():
    """Generate and plot Qini curve"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Qini Curve Visualization")
    print("=" * 70)
    
    # Generate data with clear uplift signal
    np.random.seed(42)
    n = 3000
    
    X = np.random.randn(n, 4)
    treatment = np.random.binomial(1, 0.5, n)
    
    # Strong heterogeneous effect
    true_uplift = X[:, 0] * 0.25 + X[:, 1] * 0.15
    
    p_control = 0.2
    p_treatment = np.clip(p_control + true_uplift, 0, 1)
    
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment),
        np.random.binomial(1, p_control)
    )
    
    # Get curve data
    curve = qini_curve_data(y, true_uplift, treatment, n_points=100)
    
    print(f"\nQini curve generated with {len(curve['x'])} points")
    print(f"Max uplift: {curve['y'].max():.2f}%")
    print(f"Uplift at 50%: {curve['y'][len(curve['y'])//2]:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(curve['x'], curve['y'], linewidth=2.5, color='#2E86AB')
    plt.fill_between(curve['x'], curve['y'], alpha=0.3, color='#2E86AB')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Population Targeted (%)', fontsize=11)
    plt.ylabel('Cumulative Uplift (%)', fontsize=11)
    plt.title('Qini Curve - Example', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/example_qini_curve.png', dpi=150)
    plt.close()
    print("Saved: plots/example_qini_curve.png")


# ======================== EXAMPLE 4: Treatment Imbalance ========================

def example_treatment_imbalance():
    """Demonstrate effect of treatment imbalance"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Treatment Imbalance Detection")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1000
    
    # Create imbalanced groups
    X = np.random.randn(n, 5)
    
    # Assign treatment based on features (creating confounding)
    treatment_prob = 1 / (1 + np.exp(-X[:, 0]))
    treatment = np.random.binomial(1, treatment_prob)
    
    print(f"\nTreatment assignment:")
    print(f"  Treatment: {(treatment==1).sum()} ({(treatment==1).mean()*100:.1f}%)")
    print(f"  Control: {(treatment==0).sum()} ({(treatment==0).mean()*100:.1f}%)")
    
    # Check balance
    balance = treatment_balance_check(X, treatment)
    
    print(f"\nBalance check:")
    print(f"  Average SMD: {balance['avg_smd']:.4f}")
    print(f"  Max SMD: {balance['max_smd']:.4f}")
    print(f"  Status: {balance['status']}")
    print(f"  Is Balanced: {balance['is_balanced']}")
    
    if not balance['is_balanced']:
        print("\n⚠️  WARNING: Groups are imbalanced!")
        print("    Consider using propensity score matching or weighting")


# ======================== EXAMPLE 5: Comprehensive Evaluation ========================

def example_comprehensive():
    """Use evaluate_uplift_model for complete evaluation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comprehensive Evaluation")
    print("=" * 70)
    
    # Generate realistic data
    np.random.seed(42)
    n = 5000
    
    X = np.random.randn(n, 10)
    treatment = np.random.binomial(1, 0.5, n)
    
    # Complex uplift pattern
    true_uplift = (
        X[:, 0] * 0.15 + 
        X[:, 1] * 0.10 + 
        X[:, 2] * X[:, 3] * 0.05
    )
    
    p_control = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    p_treatment = np.clip(p_control + true_uplift, 0, 1)
    
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment),
        np.random.binomial(1, p_control)
    )
    
    # Model predictions (with realistic noise)
    predicted_cate = true_uplift + np.random.randn(n) * 0.03
    
    # Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    metrics = evaluate_uplift_model(
        y_true=y,
        uplift=predicted_cate,
        treatment=treatment,
        X=X,
        k_list=[0.05, 0.1, 0.2, 0.3, 0.5]
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"\n[Qini AUC]")
    print(f"  Score: {metrics['qini_auc']:.4f}")
    
    print(f"\n[Uplift@k]")
    for k, v in sorted(metrics['uplift_at_k'].items()):
        print(f"  {k}: {v:+.2f}%")
    
    print(f"\n[Average Treatment Effect]")
    ate = metrics['ate']
    print(f"  ATE: {ate['ate']*100:+.2f}%")
    print(f"  95% CI: [{ate['ci_lower']*100:+.2f}%, {ate['ci_upper']*100:+.2f}%]")
    
    print(f"\n[Treatment Balance]")
    balance = metrics['balance']
    print(f"  Average SMD: {balance['avg_smd']:.4f}")
    print(f"  Status: {balance['status']}")


# ======================== EXAMPLE 6: Model Comparison ========================

def example_model_comparison():
    """Compare multiple models"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Model Comparison")
    print("=" * 70)
    
    np.random.seed(42)
    n = 2000
    
    X = np.random.randn(n, 5)
    treatment = np.random.binomial(1, 0.5, n)
    
    true_uplift = X[:, 0] * 0.2 + X[:, 1] * 0.1
    
    p_control = 0.25
    p_treatment = np.clip(p_control + true_uplift, 0, 1)
    
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment),
        np.random.binomial(1, p_control)
    )
    
    # Three different model predictions
    models = {
        'Perfect Model': true_uplift,
        'Good Model': true_uplift + np.random.randn(n) * 0.05,
        'Poor Model': np.random.randn(n) * 0.1,
        'Random': np.random.randn(n)
    }
    
    print("\nModel Comparison Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'Qini AUC':>12} {'Uplift@30%':>15}")
    print("-" * 70)
    
    for model_name, predictions in models.items():
        qini = qini_auc_score(y, predictions, treatment)
        u30 = uplift_at_k(y, predictions, treatment, k=0.3)
        print(f"{model_name:<20} {qini:>12.4f} {u30*100:>14.2f}%")
    
    print("-" * 70)


# ======================== RUN ALL EXAMPLES ========================

if __name__ == '__main__':
    example_basic_metrics()
    example_multiple_k()
    example_qini_curve()
    example_treatment_imbalance()
    example_comprehensive()
    example_model_comparison()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Qini AUC measures ranking quality")
    print("  • Uplift@k shows practical business value")
    print("  • Treatment balance is critical for validity")
    print("  • Use evaluate_uplift_model() for quick assessment")