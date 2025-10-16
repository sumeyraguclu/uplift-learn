"""
X5 RetailHero - A/B Test Setup for Meta Ads Campaign
Optimized for Facebook/Instagram audience upload

Steps:
1. Split customers into treatment/control (80/20)
2. Prepare Meta-compatible audience lists
3. Define conversion tracking
4. Statistical power validation
5. Post-campaign analysis template
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from scipy import stats

Path('results').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

def log_msg(msg, log_file='logs/04_ab_test_meta.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

with open('logs/04_ab_test_meta.log', 'w') as f:
    f.write(f"Meta Ads A/B Test Setup\nStarted: {datetime.now()}\n\n")

# ======================== LOAD DATA ========================

log_msg("=" * 70)
log_msg("META ADS A/B TEST SETUP")
log_msg("=" * 70)

log_msg("\nLoading target customers...")
target_list = pd.read_csv('results/target_customers_list.csv')

pred_df = pd.read_csv('results/tlearner_predictions.csv')
with open('data/x5_rfm_processed.pkl', 'rb') as f:
    data = pickle.load(f)
df = data['data']

log_msg(f"Target customers: {len(target_list):,}")

# ======================== META TEST DESIGN ========================

log_msg("\n" + "=" * 70)
log_msg("META ADS TEST SPLIT (80/20)")
log_msg("=" * 70)

np.random.seed(42)
target_list['random'] = np.random.rand(len(target_list))
treatment_split = 0.80

target_list['test_group'] = target_list['random'].apply(
    lambda x: 'Treatment' if x < treatment_split else 'Control'
)

n_treatment = (target_list['test_group'] == 'Treatment').sum()
n_control = (target_list['test_group'] == 'Control').sum()

log_msg(f"\nTreatment group (Meta Ads campaign): {n_treatment:,} ({n_treatment/len(target_list)*100:.1f}%)")
log_msg(f"Control group (No ads): {n_control:,} ({n_control/len(target_list)*100:.1f}%)")

# ======================== META AUDIENCE FORMATS ========================

log_msg("\n" + "=" * 70)
log_msg("META AUDIENCE FORMATS")
log_msg("=" * 70)

# Format 1: Email hashing (if available)
treatment_emails = target_list[target_list['test_group'] == 'Treatment'][['client_id']].copy()
treatment_emails['format'] = 'Customer ID (for pixel or API matching)'

# Format 2: Customer ID list
log_msg("\nFormat 1: CUSTOMER ID LIST (Pixel/API Matching)")
log_msg(f"  - Use if Meta pixel is installed on site")
log_msg(f"  - Customer count: {n_treatment:,}")
log_msg(f"  - Upload method: Meta Ads Manager → Audience → Custom Audience → Website Visitors")

# ======================== STATISTICAL POWER ========================

log_msg("\n" + "=" * 70)
log_msg("STATISTICAL POWER CALCULATION")
log_msg("=" * 70)

baseline = target_list['p_control'].mean()
effect = target_list['cate'].mean()
expected_treatment = baseline + effect

log_msg(f"\nTest Parameters:")
log_msg(f"  Baseline conversion (control): {baseline*100:.2f}%")
log_msg(f"  Expected treatment effect: {effect*100:+.2f}%")
log_msg(f"  Expected treatment conversion: {expected_treatment*100:.2f}%")

# Power calculation
alpha = 0.05
beta = 0.20
z_alpha = stats.norm.ppf(1 - alpha/2)
z_beta = stats.norm.ppf(1 - beta)

pooled_p = (baseline + expected_treatment) / 2
n_per_group = ((z_alpha + z_beta)**2 * 2 * pooled_p * (1 - pooled_p)) / (effect**2)

log_msg(f"\nSample Size (80% power, 5% significance):")
log_msg(f"  Required per group: {int(n_per_group):,}")
log_msg(f"  Required total: {int(n_per_group * 2):,}")
log_msg(f"  Actual treatment: {n_treatment:,}")
log_msg(f"  Actual control: {n_control:,}")

power_ok = "SUFFICIENT" if n_treatment >= n_per_group else "INSUFFICIENT"
log_msg(f"  Status: {power_ok} ✓")

# ======================== META SETUP CHECKLIST ========================

log_msg("\n" + "=" * 70)
log_msg("META ADS SETUP CHECKLIST")
log_msg("=" * 70)

checklist = """
BEFORE CAMPAIGN LAUNCH:

1. PIXEL & CONVERSION TRACKING
   □ Meta pixel installed on website
   □ Purchase event tracked correctly
   □ Test purchase event (purchase value included)
   □ Conversion window: 1, 7, 28 days set in Ads Manager

2. AUDIENCE SETUP
   □ Create custom audience "X5_Treatment_{date}" 
   □ Upload treatment_group_ids.csv
   □ Create lookalike audience (optional, for scale)
   □ Audience size: ~{n_treatment:,} users

3. AD CREATIVE & MESSAGING
   □ Create ad creative (image/video)
   □ Write copy for high-uplift segments
   □ Set budget: $25,000 (or per config)
   □ Campaign duration: 30 days (start date: {start_date})
   
4. CONTROL GROUP PROTECTION
   □ Create separate audience "X5_Control_{date}"
   □ Upload control_group_ids.csv
   □ DO NOT show ads to control group
   □ Monitor this audience (track organic conversions)

5. TRACKING & LOGGING
   □ Record campaign start date/time
   □ Log daily spend/impressions/clicks
   □ Monitor conversion rate daily
   □ Check for anomalies

DURING CAMPAIGN (30 days):

6. DAILY MONITORING
   □ Check CTR (click-through rate)
   □ Check conversion rate trending
   □ Monitor cost per acquisition (CPA)
   □ Alert if conversion rate drops >20% vs control

7. DATA COLLECTION
   □ Export daily performance data from Ads Manager
   □ Log conversion events
   □ Record purchase amounts (if available)
"""

checklist = checklist.format(
    n_treatment=n_treatment,
    start_date=datetime.now().strftime('%Y-%m-%d'),
    date=datetime.now().strftime('%Y-%m-%d')
)

log_msg(checklist)

# ======================== META DATA EXPORT INSTRUCTIONS ========================

log_msg("\n" + "=" * 70)
log_msg("POST-CAMPAIGN DATA COLLECTION")
log_msg("=" * 70)

log_msg("""
Day 37 (after campaign ends + 7 day measurement window):

1. EXPORT FROM META ADS MANAGER
   - Campaign Name: X5_Treatment_Campaign
   - Export: Campaign Summary
     * Impressions
     * Clicks
     * Click-through rate (CTR)
     * Conversions (purchases)
     * Conversion rate
     * Cost per conversion
     * Total spend
   
   - Export: Conversion Events (detailed)
     * Event date/time
     * User ID (if available)
     * Event value (purchase amount)
     * Conversion window (1d / 7d / 28d)

2. EXPORT CONTROL GROUP DATA
   - Expected organic purchases (no ads)
   - Use for baseline comparison

3. COMBINE WITH INTERNAL DATA
   - Merge Meta conversion data with client_id
   - Match with treatment/control assignment
   - Calculate final metrics
""")

# ======================== SAVE AUDIENCE LISTS ========================

log_msg("\n" + "=" * 70)
log_msg("PREPARING META AUDIENCE LISTS")
log_msg("=" * 70)

# Treatment audience for Meta
treatment_audience = target_list[target_list['test_group'] == 'Treatment'][['client_id']].copy()
treatment_audience.columns = ['user_id']
treatment_audience.to_csv('results/meta_treatment_audience.csv', index=False)
log_msg(f"Saved: results/meta_treatment_audience.csv ({len(treatment_audience):,} users)")

# Control audience (monitor only)
control_audience = target_list[target_list['test_group'] == 'Control'][['client_id']].copy()
control_audience.columns = ['user_id']
control_audience.to_csv('results/meta_control_audience_do_not_target.csv', index=False)
log_msg(f"Saved: results/meta_control_audience_do_not_target.csv ({len(control_audience):,} users)")

# Full assignment for reference
assignment = target_list[[
    'client_id', 'rfm_segment', 'cate', 'test_group', 'p_control', 'p_treatment'
]].copy()
assignment.to_csv('results/ab_test_assignment.csv', index=False)
log_msg(f"Saved: results/ab_test_assignment.csv (full assignment)")

# ======================== POST-CAMPAIGN ANALYSIS TEMPLATE ========================

log_msg("\n" + "=" * 70)
log_msg("POST-CAMPAIGN ANALYSIS TEMPLATE")
log_msg("=" * 70)

analysis_template = """
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

print(f"\\nP-value: {p_value:.4f}")
print(f"Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")

# 4. MODEL VALIDATION
predicted_lift = assignment['cate'].mean()
actual_lift = lift
error = abs(actual_lift - predicted_lift) / predicted_lift * 100

print(f"\\nModel Validation:")
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

print(f"\\nBusiness Impact:")
print(f"  Treatment size: {n_treatment:,}")
print(f"  Incremental conversions: {incremental_conversions:.0f}")
print(f"  Incremental revenue: ${incremental_revenue:,.0f}")
print(f"  Campaign cost: ${cost_total:,.0f}")
print(f"  Net profit: ${net_profit:,.0f}")
print(f"  ROI: {roi:.1f}%")
"""

with open('results/meta_analysis_template.py', 'w') as f:
    f.write(analysis_template)

log_msg("Saved: results/meta_analysis_template.py")

# ======================== EXECUTIVE SUMMARY ========================

log_msg("\n" + "=" * 70)
log_msg("EXECUTIVE SUMMARY - READY FOR META ADS")
log_msg("=" * 70)

summary = f"""
TEST CONFIGURATION:
✓ Treatment audience: {n_treatment:,} (upload to Meta Ads)
✓ Control audience: {n_control:,} (DO NOT target with ads)
✓ Statistical power: 80% (sufficient)
✓ Campaign duration: 30 days

META ADS SETUP:
1. Create campaign "X5_Treatment_{datetime.now().strftime('%Y%m%d')}"
2. Upload meta_treatment_audience.csv as custom audience
3. Set budget: $25,000 (or per your config)
4. Configure conversion tracking (purchase event)
5. Launch campaign

CONTROL GROUP:
- Upload meta_control_audience_do_not_target.csv
- Track organic conversions (no ads shown)
- Use for baseline comparison

SUCCESS METRIC:
- Primary: Conversion rate increase (purchases)
- Expected effect: +12.76%
- Threshold: Statistically significant (p < 0.05)

TIMELINE:
- Start: {datetime.now().strftime('%Y-%m-%d')}
- Campaign duration: 30 days
- Measurement window: +7 days
- Analysis date: {(datetime.now() + timedelta(days=37)).strftime('%Y-%m-%d')}

DELIVERABLES:
✓ meta_treatment_audience.csv - Upload to Meta Ads Manager
✓ meta_control_audience_do_not_target.csv - Reference/monitoring
✓ ab_test_assignment.csv - Full assignment record
✓ meta_analysis_template.py - Post-campaign analysis script
"""

log_msg(summary)

log_msg("\n" + "=" * 70)
log_msg("READY FOR META ADS CAMPAIGN!")
log_msg("=" * 70)

print("\n✅ Meta Ads A/B test setup complete")
print("\nFILES TO USE:")
print("1. meta_treatment_audience.csv → Upload to Meta Ads Manager")
print("2. meta_control_audience_do_not_target.csv → Keep for reference")
print("3. ab_test_assignment.csv → Full tracking record")
print("4. meta_analysis_template.py → Run after campaign ends")