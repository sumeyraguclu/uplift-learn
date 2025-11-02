"""
Optimization Module Usage Examples
Demonstrates how to use src.optimize functions
"""

import numpy as np
import pandas as pd
from src.optimize import (
    greedy_optimizer,
    roi_threshold_optimizer,
    top_k_optimizer,
    compare_strategies,
    calculate_campaign_metrics,
    optimize_with_constraints
)


# ======================== EXAMPLE 1: Greedy Optimization ========================

def example_greedy():
    """Budget-constrained greedy optimization"""
    print("=" * 70)
    print("EXAMPLE 1: Greedy Optimization (Budget-Constrained)")
    print("=" * 70)
    
    # Generate synthetic uplift scores
    np.random.seed(42)
    n_customers = 10000
    uplift = np.random.beta(2, 5, n_customers) * 0.15  # 0-15% uplift
    
    # Campaign parameters
    margin = 50.0  # $50 per conversion
    contact_cost = 0.5  # $0.5 per contact
    budget = 2000.0  # $2k budget
    
    # Optimize
    result = greedy_optimizer(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        budget=budget
    )
    
    print(f"\nBudget: ${budget:,.0f}")
    print(f"Max customers (budget/cost): {int(budget/contact_cost):,}")
    print(f"\nResults:")
    print(f"  Selected: {result['n_selected']:,} customers")
    print(f"  Total cost: ${result['total_cost']:,.2f}")
    print(f"  Budget used: {result['budget_used']/budget*100:.1f}%")
    print(f"  Expected revenue: ${result['expected_revenue']:,.2f}")
    print(f"  Expected profit: ${result['expected_profit']:,.2f}")
    print(f"  ROI: {result['roi_pct']:.1f}%")
    print(f"  Avg uplift: {result['avg_uplift']:.2%}")


# ======================== EXAMPLE 2: ROI Threshold ========================

def example_roi_threshold():
    """Select all profitable customers"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: ROI Threshold Optimization")
    print("=" * 70)
    
    np.random.seed(42)
    n_customers = 5000
    
    # Heterogeneous uplift
    uplift = np.random.randn(n_customers) * 0.03 + 0.05  # Mean 5%, std 3%
    uplift = np.clip(uplift, -0.05, 0.20)
    
    margin = 40.0
    contact_cost = 0.8
    
    # Different ROI thresholds
    thresholds = [0.0, 0.5, 1.0, 2.0]
    
    print("\nROI Threshold Comparison:")
    print("-" * 70)
    print(f"{'Threshold':<15} {'Selected':<12} {'Cost':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 70)
    
    for thresh in thresholds:
        result = roi_threshold_optimizer(
            uplift=uplift,
            margin=margin,
            contact_cost=contact_cost,
            min_roi=thresh
        )
        
        print(f"{'≥'+str(int(thresh*100))+'%':<15} "
              f"{result['n_selected']:<12,} "
              f"${result['total_cost']:<11,.0f} "
              f"${result['expected_profit']:<11,.0f} "
              f"{result['roi_pct']:<10.1f}%")


# ======================== EXAMPLE 3: Top-K Strategies ========================

def example_top_k():
    """Compare different top-k strategies"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Top-K Targeting")
    print("=" * 70)
    
    np.random.seed(42)
    n_customers = 8000
    
    # Strong positive uplift for top segment
    uplift = np.concatenate([
        np.random.beta(4, 2, 800) * 0.15,  # High uplift
        np.random.beta(2, 3, 2200) * 0.08,  # Medium uplift
        np.random.beta(1, 4, 5000) * 0.03   # Low uplift
    ])
    np.random.shuffle(uplift)
    
    margin = 60.0
    contact_cost = 0.6
    
    # Different k values
    k_values = [0.05, 0.10, 0.20, 0.30, 0.50, 1000]  # Mix of fractions and absolute
    
    print("\nTop-K Strategy Comparison:")
    print("-" * 70)
    
    for k in k_values:
        result = top_k_optimizer(
            uplift=uplift,
            k=k,
            margin=margin,
            contact_cost=contact_cost
        )
        
        k_label = f"{k*100:.0f}%" if isinstance(k, float) and k < 1 else f"{k}"
        print(f"\nTop-{k_label}:")
        print(f"  Customers: {result['n_selected']:,}")
        print(f"  Avg uplift: {result['avg_uplift']:.2%}")
        print(f"  ROI: {result['roi_pct']:.1f}%")
        print(f"  Profit: ${result['expected_profit']:,.0f}")


# ======================== EXAMPLE 4: Strategy Comparison ========================

def example_strategy_comparison():
    """Compare all strategies at once"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comprehensive Strategy Comparison")
    print("=" * 70)
    
    np.random.seed(42)
    n_customers = 12000
    
    # Realistic uplift distribution
    uplift = np.random.gamma(2, 0.02, n_customers)  # Right-skewed
    uplift = np.clip(uplift, 0, 0.25)
    
    margin = 55.0
    contact_cost = 0.7
    budget = 5000.0
    
    # Compare all strategies
    comparison = compare_strategies(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        budget=budget,
        k_values=[0.1, 0.2, 0.3],
        roi_thresholds=[0.0, 0.5, 1.0]
    )
    
    # Sort by ROI
    comparison_sorted = comparison.sort_values('roi_pct', ascending=False)
    
    print("\nAll Strategies Ranked by ROI:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'N':<10} {'Cost':<12} {'Revenue':<12} {'Profit':<12} {'ROI':<10}")
    print("-" * 80)
    
    for _, row in comparison_sorted.iterrows():
        print(f"{row['strategy']:<20} "
              f"{row['n_selected']:<10,.0f} "
              f"${row['cost']:<11,.0f} "
              f"${row['revenue']:<11,.0f} "
              f"${row['profit']:<11,.0f} "
              f"{row['roi_pct']:<10.1f}%")
    
    print("\n" + "=" * 80)
    best = comparison_sorted.iloc[0]
    print(f"RECOMMENDED: {best['strategy']}")
    print(f"  Expected ROI: {best['roi_pct']:.1f}%")
    print(f"  Net Profit: ${best['profit']:,.0f}")


# ======================== EXAMPLE 5: Multi-Constraint Optimization ========================

def example_multi_constraint():
    """Optimize with multiple constraints"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Constraint Optimization")
    print("=" * 70)
    
    np.random.seed(42)
    n_customers = 15000
    
    uplift = np.random.exponential(0.04, n_customers)
    uplift = np.clip(uplift, 0, 0.20)
    
    margin = 45.0
    contact_cost = 0.55
    
    # Scenario 1: Budget only
    print("\nScenario 1: Budget constraint only")
    result1 = optimize_with_constraints(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        budget=3000
    )
    print(f"  Selected: {result1['n_selected']:,}")
    print(f"  ROI: {result1['roi_pct']:.1f}%")
    
    # Scenario 2: ROI only
    print("\nScenario 2: ROI constraint only (≥50%)")
    result2 = optimize_with_constraints(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        min_roi=0.5
    )
    print(f"  Selected: {result2['n_selected']:,}")
    print(f"  ROI: {result2['roi_pct']:.1f}%")
    print(f"  Cost: ${result2['total_cost']:,.0f}")
    
    # Scenario 3: Budget + ROI
    print("\nScenario 3: Budget + ROI constraints")
    result3 = optimize_with_constraints(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        budget=3000,
        min_roi=0.5
    )
    print(f"  Selected: {result3['n_selected']:,}")
    print(f"  ROI: {result3['roi_pct']:.1f}%")
    print(f"  Cost: ${result3['total_cost']:,.0f}")
    
    # Scenario 4: All constraints
    print("\nScenario 4: Budget + ROI + Max customers")
    result4 = optimize_with_constraints(
        uplift=uplift,
        margin=margin,
        contact_cost=contact_cost,
        budget=5000,
        min_roi=0.3,
        max_customers=3000
    )
    print(f"  Selected: {result4['n_selected']:,}")
    print(f"  ROI: {result4['roi_pct']:.1f}%")
    print(f"  Cost: ${result4['total_cost']:,.0f}")


# ======================== EXAMPLE 6: Campaign Metrics ========================

def example_campaign_metrics():
    """Calculate comprehensive campaign metrics"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Campaign Metrics Calculation")
    print("=" * 70)
    
    np.random.seed(42)
    n_customers = 5000
    
    # Predicted uplift
    uplift = np.random.beta(2, 5, n_customers) * 0.12
    
    # Actual treatment (for post-campaign analysis)
    treatment = np.random.binomial(1, 0.8, n_customers)  # 80% treated
    
    # Select profitable customers
    margin = 50.0
    contact_cost = 0.5
    
    result = roi_threshold_optimizer(uplift, margin, contact_cost, min_roi=0.0)
    selected_mask = result['selected_mask']
    
    # Calculate metrics
    metrics = calculate_campaign_metrics(
        uplift=uplift,
        treatment=treatment,
        margin=margin,
        contact_cost=contact_cost,
        selected_mask=selected_mask
    )
    
    print(f"\nCampaign Metrics:")
    print(f"  Total customers: {metrics['n_total']:,}")
    print(f"  Treated: {metrics['n_treated']:,} ({metrics['treatment_ratio']*100:.0f}%)")
    print(f"  Control: {metrics['n_control']:,}")
    print(f"\nFinancial:")
    print(f"  Total cost: ${metrics['total_cost']:,.2f}")
    print(f"  Expected revenue: ${metrics['expected_revenue']:,.2f}")
    print(f"  Expected profit: ${metrics['expected_profit']:,.2f}")
    print(f"  ROI: {metrics['roi_pct']:.1f}%")
    print(f"\nUplift:")
    print(f"  Average uplift: {metrics['avg_uplift']:.2%}")
    print(f"  Total expected incremental conversions: {metrics['total_expected_uplift']:.1f}")


# ======================== EXAMPLE 7: Real-World Scenario ========================

def example_real_world():
    """Realistic e-commerce campaign"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Real-World E-commerce Campaign")
    print("=" * 70)
    
    # Simulate customer base
    np.random.seed(42)
    n_customers = 50000
    
    # Create customer segments with different uplift
    segment_sizes = [5000, 10000, 15000, 20000]  # VIP, High, Medium, Low
    segment_uplifts = [
        np.random.beta(5, 2, segment_sizes[0]) * 0.20,  # VIP: high uplift
        np.random.beta(3, 3, segment_sizes[1]) * 0.12,  # High: medium uplift
        np.random.beta(2, 5, segment_sizes[2]) * 0.06,  # Medium: low uplift
        np.random.beta(1, 6, segment_sizes[3]) * 0.02   # Low: very low uplift
    ]
    
    uplift = np.concatenate(segment_uplifts)
    np.random.shuffle(uplift)
    
    # E-commerce parameters
    avg_order_value = 75.0  # $75 AOV
    email_cost = 0.02  # $0.02 per email
    monthly_budget = 5000.0  # $5k/month
    
    print(f"\nCustomer Base: {n_customers:,}")
    print(f"Average Order Value: ${avg_order_value:.2f}")
    print(f"Email Cost: ${email_cost:.2f}")
    print(f"Monthly Budget: ${monthly_budget:,.0f}")
    
    # Compare strategies
    print("\n" + "-" * 70)
    print("STRATEGY EVALUATION")
    print("-" * 70)
    
    comparison = compare_strategies(
        uplift=uplift,
        margin=avg_order_value,
        contact_cost=email_cost,
        budget=monthly_budget,
        k_values=[0.05, 0.10, 0.20],
        roi_thresholds=[0.0, 1.0, 2.0]
    )
    
    for _, row in comparison.sort_values('roi_pct', ascending=False).head(5).iterrows():
        print(f"\n{row['strategy']}:")
        print(f"  Emails sent: {row['n_selected']:,}")
        print(f"  Cost: ${row['cost']:,.2f}")
        print(f"  Expected revenue: ${row['revenue']:,.2f}")
        print(f"  Net profit: ${row['profit']:,.2f}")
        print(f"  ROI: {row['roi_pct']:.0f}%")
    
    best = comparison.sort_values('roi_pct', ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print(f"RECOMMENDATION: {best['strategy']}")
    print(f"Expected monthly profit: ${best['profit']:,.0f} (ROI: {best['roi_pct']:.0f}%)")


# ======================== RUN ALL EXAMPLES ========================

if __name__ == '__main__':
    example_greedy()
    example_roi_threshold()
    example_top_k()
    example_strategy_comparison()
    example_multi_constraint()
    example_campaign_metrics()
    example_real_world()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Greedy optimization is optimal for fixed costs")
    print("  • ROI thresholds ensure profitability")
    print("  • Top-k strategies are simple and effective")
    print("  • compare_strategies() helps find the best approach")
    print("  • Multi-constraint optimization handles complex scenarios")