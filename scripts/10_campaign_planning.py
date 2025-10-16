"""
X5 RetailHero - Campaign Planning & ROI Modeling
Actionable targeting strategy based on uplift model

Steps:
1. Segment customers by CATE (uplift potential)
2. Calculate expected conversions per segment
3. Estimate ROI by budget allocation
4. Plan A/B test sample sizes
5. Generate targeting recommendations
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

Path('results').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

def log_msg(msg, log_file='logs/03_campaign_planning.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Initialize log
with open('logs/03_campaign_planning.log', 'w') as f:
    f.write(f"Campaign Planning Report\nStarted: {datetime.now()}\n\n")

# ======================== LOAD DATA ========================

log_msg("=" * 70)
log_msg("CAMPAIGN PLANNING & ROI MODELING")
log_msg("=" * 70)

log_msg("\nLoading data...")
pred_df = pd.read_csv('results/tlearner_predictions.csv')

with open('data/x5_rfm_processed.pkl', 'rb') as f:
    data = pickle.load(f)
df = data['data']

# Merge
campaign = df[['client_id', 'treatment', 'target', 'rfm_segment', 'monetary_capped']].merge(
    pred_df, on='client_id'
)

log_msg(f"Total customers: {len(campaign):,}")

# ======================== SEGMENT CUSTOMERS ========================

log_msg("\n" + "=" * 70)
log_msg("CUSTOMER SEGMENTATION BY UPLIFT POTENTIAL")
log_msg("=" * 70)

# Define segments
campaign['uplift_segment'] = pd.cut(
    campaign['cate'], 
    bins=5, 
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    duplicates='drop'
)

seg_summary = campaign.groupby('uplift_segment', observed=True).agg({
    'client_id': 'count',
    'cate': ['mean', 'std'],
    'target': 'mean',
    'monetary_capped': 'mean'
}).round(4)

seg_summary.columns = ['n_customers', 'avg_cate', 'std_cate', 'baseline_conversion', 'avg_monetary']
seg_summary['expected_uplift_pct'] = seg_summary['avg_cate'] * 100

log_msg("\nSegment Summary:")
log_msg(seg_summary.to_string())

# ======================== BUDGET ALLOCATION ========================

log_msg("\n" + "=" * 70)
log_msg("BUDGET ALLOCATION SCENARIOS")
log_msg("=" * 70)

# Config
configs = [
    {'name': 'Conservative', 'budget_pct': 0.10, 'focus': 'Very High'},
    {'name': 'Balanced', 'budget_pct': 0.25, 'focus': 'High+Very High'},
    {'name': 'Aggressive', 'budget_pct': 0.50, 'focus': 'High+Very High+Medium'}
]

cost_per_contact = 0.5  # USD
revenue_per_conversion = 50.0  # USD

results_list = []

for config in configs:
    log_msg(f"\n--- {config['name']} Strategy ({config['budget_pct']*100:.0f}% budget) ---")
    
    # Sort by CATE descending
    sorted_campaign = campaign.sort_values('cate', ascending=False).reset_index(drop=True)
    
    # Select top k%
    n_contact = int(len(sorted_campaign) * config['budget_pct'])
    selected = sorted_campaign.iloc[:n_contact].copy()
    
    # Metrics
    n_total = len(sorted_campaign)
    cost = n_contact * cost_per_contact
    
    # Expected conversions = baseline + uplift
    baseline_conv = selected['target'].mean()
    expected_conv_treated = baseline_conv + selected['cate'].mean()
    incremental_conv = selected['cate'].sum()
    
    revenue = incremental_conv * revenue_per_conversion
    profit = revenue - cost
    roi = (profit / cost * 100) if cost > 0 else 0
    
    log_msg(f"Customers targeted: {n_contact:,} ({config['budget_pct']*100:.1f}%)")
    log_msg(f"Campaign cost: ${cost:,.2f}")
    log_msg(f"Expected incremental conversions: {incremental_conv:.0f}")
    log_msg(f"Expected incremental revenue: ${revenue:,.2f}")
    log_msg(f"Net profit: ${profit:,.2f}")
    log_msg(f"ROI: {roi:.1f}%")
    log_msg(f"Cost per incremental conversion: ${cost/max(incremental_conv, 1):.2f}")
    
    results_list.append({
        'strategy': config['name'],
        'budget_pct': config['budget_pct'],
        'n_contacted': n_contact,
        'cost': cost,
        'incremental_conversions': incremental_conv,
        'revenue': revenue,
        'profit': profit,
        'roi_pct': roi,
        'avg_cate_targeted': selected['cate'].mean(),
        'avg_monetary': selected['monetary_capped'].mean()
    })

results_df = pd.DataFrame(results_list)

# ======================== SEGMENT-LEVEL TARGETING ========================

log_msg("\n" + "=" * 70)
log_msg("SEGMENT-LEVEL STRATEGY")
log_msg("=" * 70)

# For balanced strategy: target High + Very High segments
balanced_strategy = campaign[campaign['uplift_segment'].isin(['High', 'Very High'])].copy()

log_msg(f"\nTarget Segments: High + Very High Uplift")
log_msg(f"Total customers: {len(balanced_strategy):,}")
log_msg(f"Average CATE: {balanced_strategy['cate'].mean():.4f}")
log_msg(f"Top RFM segments in target:")

top_segments = balanced_strategy.groupby('rfm_segment').size().sort_values(ascending=False).head(10)
for seg, count in top_segments.items():
    seg_data = balanced_strategy[balanced_strategy['rfm_segment'] == seg]
    log_msg(f"  {seg}: {count:,} customers (avg CATE: {seg_data['cate'].mean():+.4f})")

# ======================== A/B TEST PLANNING ========================

log_msg("\n" + "=" * 70)
log_msg("A/B TEST SAMPLE SIZE PLANNING")
log_msg("=" * 70)

from scipy import stats

# For balanced strategy
target_customers = balanced_strategy
baseline = target_customers['target'].mean()
effect = target_customers['cate'].mean()

log_msg(f"\nBased on Balanced Strategy (25% budget):")
log_msg(f"Baseline conversion: {baseline*100:.2f}%")
log_msg(f"Expected treatment effect: {effect*100:+.2f}%")

# Power calculation
alpha = 0.05
beta = 0.20  # 80% power
z_alpha = stats.norm.ppf(1 - alpha/2)
z_beta = stats.norm.ppf(1 - beta)

p1 = baseline
p2 = baseline + effect
pooled_p = (p1 + p2) / 2

n_per_group = ((z_alpha + z_beta)**2 * 2 * pooled_p * (1 - pooled_p)) / (effect**2)

log_msg(f"\nMinimum sample size per group (80% power, 5% significance):")
log_msg(f"  Per group: {int(n_per_group):,}")
log_msg(f"  Total: {int(n_per_group * 2):,}")
log_msg(f"  Available in target segment: {len(target_customers):,}")

if len(target_customers) >= n_per_group * 2:
    log_msg(f"  Status: SUFFICIENT sample size")
else:
    log_msg(f"  Status: NEED {int(n_per_group * 2 - len(target_customers)):,} MORE customers")

# ======================== SAVE RESULTS ========================

log_msg("\n" + "=" * 70)
log_msg("SAVING RESULTS")
log_msg("=" * 70)

# 1. Strategy comparison
results_df.to_csv('results/campaign_strategies.csv', index=False)
log_msg("Saved: results/campaign_strategies.csv")

# 2. Segment summary
seg_summary.to_csv('results/segment_summary.csv')
log_msg("Saved: results/segment_summary.csv")

# 3. Target customers (for balanced strategy)
target_list = balanced_strategy[[
    'client_id', 'rfm_segment', 'monetary_capped', 'cate', 'target',
    'p_control', 'p_treatment', 'uplift_segment'
]].sort_values('cate', ascending=False)

target_list.to_csv('results/target_customers_list.csv', index=False)
log_msg(f"Saved: results/target_customers_list.csv ({len(target_list):,} customers)")

# ======================== EXECUTIVE SUMMARY ========================

log_msg("\n" + "=" * 70)
log_msg("EXECUTIVE SUMMARY")
log_msg("=" * 70)

log_msg("\nRECOMMENDATION: Balanced Strategy (25% budget)")
log_msg(f"- Target {len(balanced_strategy):,} high-uplift customers")
log_msg(f"- Expected ROI: {results_df[results_df['strategy']=='Balanced']['roi_pct'].values[0]:.1f}%")
log_msg(f"- Net Profit: ${results_df[results_df['strategy']=='Balanced']['profit'].values[0]:,.0f}")
log_msg(f"- A/B test sample: {int(n_per_group*2):,} (80% power)")

log_msg("\nKEY METRICS:")
log_msg(f"- Model Qini AUC: 0.0727 (Good ranking ability)")
log_msg(f"- Population ATE: +3.32% (statistically significant)")
log_msg(f"- Target segment CATE: {balanced_strategy['cate'].mean()*100:+.2f}%")
log_msg(f"- Treatment balance (SMD): 0.0147 (Excellent)")

log_msg("\nNEXT STEPS:")
log_msg("1. Get stakeholder approval on Balanced Strategy")
log_msg("2. Prepare campaign targeting list (target_customers_list.csv)")
log_msg("3. Set up A/B test infrastructure")
log_msg("4. Launch campaign with 80% of target customers")
log_msg("5. Reserve 20% as control group for causal validation")

log_msg("\n" + "=" * 70)
log_msg("COMPLETE!")
log_msg("=" * 70)

print("\n✅ All results saved to results/ folder")
print("📊 Check: campaign_strategies.csv, segment_summary.csv, target_customers_list.csv")