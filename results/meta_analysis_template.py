
# Meta Ads A/B Test Analysis (Run on Day 37)

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
import numpy as np

# 1. LOAD META DATA & ASSIGNMENT
meta_data = pd.read_csv('meta_campaign_results.csv')  # From Ads Manager export
assignment = pd.read_csv('results/ab_test_assignment.csv')

# Merge
data = assignment.merge(meta_data, on='client_id', how='left')

# 2. PRIMARY METRIC: Conversion Rate
conversions = data.groupby('test_group').agg({
    'purchase_made': ['sum', 'count', 'mean']
}).round(4)

conv_treatment = data[data['test_group']=='Treatment']['purchase_made'].mean()
conv_control = data[data['test_group']=='Control']['purchase_made'].mean()
lift = conv_treatment - conv_control

print(f"Conversion Rate (Treatment): {conv_treatment*100:.2f}%")
print(f"Conversion Rate (Control): {conv_control*100:.2f}%")
print(f"Absolute Lift: {lift*100:+.2f} percentage points")
print(f"Relative Lift: {(lift / conv_control)*100:+.2f}%")

# 3. STATISTICAL SIGNIFICANCE
contingency = pd.crosstab(data['test_group'], data['purchase_made'])
chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"\nP-value: {p_value:.4f}")
print(f"Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")

# 4. MODEL VALIDATION
predicted_lift = assignment['cate'].mean()
actual_lift = lift
error = abs(actual_lift - predicted_lift) / predicted_lift * 100

print(f"\nModel Validation:")
print(f"  Predicted uplift: {predicted_lift*100:.2f}%")
print(f"  Actual uplift: {actual_lift*100:.2f}%")
print(f"  Prediction error: {error:.1f}%")

# 5. BUSINESS METRICS
n_treatment = len(data[data['test_group']=='Treatment'])
cost_total = 25000  # Campaign budget
incremental_conversions = lift * n_treatment
incremental_revenue = incremental_conversions * 50  # $50 per purchase
net_profit = incremental_revenue - cost_total
roi = (net_profit / cost_total) * 100

print(f"\nBusiness Impact:")
print(f"  Treatment size: {n_treatment:,}")
print(f"  Incremental conversions: {incremental_conversions:.0f}")
print(f"  Incremental revenue: ${incremental_revenue:,.0f}")
print(f"  Campaign cost: ${cost_total:,.0f}")
print(f"  Net profit: ${net_profit:,.0f}")
print(f"  ROI: {roi:.1f}%")
