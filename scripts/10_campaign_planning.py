"""
X5 RetailHero - Campaign Planning (Refactored)
Uses centralized src.optimize module

Strategy comparison and ROI/Budget constrained targeting
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

# Import from src modules
from src.optimize import (
    greedy_optimizer,
    roi_threshold_optimizer,
    top_k_optimizer,
    compare_strategies,
    calculate_campaign_metrics,
    optimize_with_constraints,  # kısıtlı optimizasyon
)
from src.config import get_config

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# Config & Dirs
# ---------------------------------------------------------------------
config = get_config()
config.ensure_dirs()
Path('logs').mkdir(exist_ok=True)

def log_msg(msg, log_file='logs/10_campaign_planning.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Initialize log
with open('logs/10_campaign_planning.log', 'w', encoding='utf-8') as f:
    f.write(f"Campaign Planning (Refactored)\nStarted: {datetime.now()}\n\n")

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
log_msg("=" * 70)
log_msg("CAMPAIGN PLANNING & ROI OPTIMIZATION")
log_msg("=" * 70)
log_msg("Using src.optimize module")
log_msg(f"Config: margin=${config.campaign.margin}, budget=${config.campaign.budget:,.0f}, "
        f"contact_cost=${config.campaign.contact_cost:.2f}, "
        f"min_roi={getattr(config.campaign, 'min_roi', 0.5):.2f}")

log_msg("\nLoading data...")

# Tercihen config.paths.calibrated_predictions, yoksa results/final_cate.csv
pred_path = getattr(getattr(config, 'paths', object()), 'calibrated_predictions', None)
if pred_path is None:
    pred_path = 'results/final_cate.csv'
pred_df = pd.read_csv(pred_path)

# --- Tek ve nihai 'cate' kolonunu garanti et ---
if 'cate_calibrated' in pred_df.columns:
    keep_cols = ['client_id', 'cate_calibrated'] + [c for c in ['p_control', 'p_treatment'] if c in pred_df.columns]
    pred_df = pred_df[keep_cols].copy()
    pred_df = pred_df.rename(columns={'cate_calibrated': 'cate'})
elif 'cate' in pred_df.columns:
    keep_cols = ['client_id', 'cate'] + [c for c in ['p_control', 'p_treatment'] if c in pred_df.columns]
    pred_df = pred_df[keep_cols].copy()
else:
    raise ValueError(f"{pred_path} dosyasında 'cate_calibrated' veya 'cate' kolonu bulunmalı.")

# RFM & ana veri
with open(config.paths.rfm_data, 'rb') as f:
    data = pickle.load(f)
df = data['data']

# Merge
campaign = df[[
    'client_id', 'treatment', 'target',
    'rfm_segment', 'r_score', 'f_score', 'm_score', 'rfm_score',
    'monetary_capped', 'recency', 'frequency', 'aov'
]].merge(pred_df, on='client_id', how='inner')

# Yinelenen kolon adlarını temizle (sonuncuyu tut)
campaign = campaign.loc[:, ~campaign.columns.duplicated(keep='last')]

# Emniyet: tek 'cate' kolonu
cate_cols = [c for c in campaign.columns if c == 'cate']
assert len(cate_cols) == 1, f"Beklenmedik cate kolon sayısı: {cate_cols}"
campaign['cate'] = pd.to_numeric(campaign['cate'], errors='coerce')

if 'rfm_segment' not in campaign.columns:
    campaign['rfm_segment'] = 'UNKNOWN'

log_msg(f"Total customers: {len(campaign):,}")

# ---------------------------------------------------------------------
# STRATEGY COMPARISON (sadece karar desteği)
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("STRATEGY COMPARISON (using src.optimize)")
log_msg("=" * 70)

uplift = campaign['cate'].to_numpy()  # 1D
indices = np.arange(uplift.shape[0])

comparison_df = compare_strategies(
    uplift=uplift,
    margin=config.campaign.margin,
    contact_cost=config.campaign.contact_cost,
    budget=config.campaign.budget,
    k_values=[0.1, 0.2, 0.3, 0.5],
    roi_thresholds=[0.0, 0.5, 1.0],
    indices=indices
)

log_msg("\nStrategy Comparison Results:")
log_msg("-" * 70)
for _, row in comparison_df.sort_values('roi_pct', ascending=False).iterrows():
    log_msg(f"{row['strategy']:<20} | "
            f"n={row['n_selected']:>6,.0f} | "
            f"Cost=${row['cost']:>8,.0f} | "
            f"Profit=${row['profit']:>8,.0f} | "
            f"ROI={row['roi_pct']:>6.1f}%")

comparison_df.to_csv('results/campaign_strategies_comparison.csv', index=False)
log_msg("\nSaved: results/campaign_strategies_comparison.csv")

# Kıyaslamada en yüksek ROI'yi loglayalım (bilgi amaçlı)
best_idx = comparison_df['roi_pct'].idxmax()
best_strategy_row = comparison_df.loc[best_idx]
log_msg("\n" + "=" * 70)
log_msg("RECOMMENDED STRATEGY (from comparison, info only)")
log_msg("=" * 70)
log_msg(f"{best_strategy_row['strategy']}: "
        f"n={best_strategy_row['n_selected']:,}, "
        f"Cost=${best_strategy_row['cost']:,.0f}, "
        f"Profit=${best_strategy_row['profit']:,.0f}, "
        f"ROI={best_strategy_row['roi_pct']:.1f}%")

# ---------------------------------------------------------------------
# GENERATE TARGET LISTS (Optimized with ROI & Budget constraints)
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("GENERATING TARGET LISTS (Optimized with Constraints)")
log_msg("=" * 70)

min_roi_constraint = float(getattr(config.campaign, 'min_roi', 0.0))
budget_constraint = float(getattr(config.campaign, 'budget', float('inf')))

constraint_line = f"ROI ≥ {min_roi_constraint*100:.0f}% AND Budget ≤ ${budget_constraint:,.0f}"
method_label = "src.optimize.optimize_with_constraints"
log_msg(f"Applying constraints: {constraint_line}")

final_result = optimize_with_constraints(
    uplift=uplift,
    margin=config.campaign.margin,
    contact_cost=config.campaign.contact_cost,
    budget=budget_constraint,
    min_roi=min_roi_constraint,
    max_customers=None,
    indices=indices
)

sel_idx = np.asarray(final_result['selected_indices'], dtype=int).reshape(-1)
target_list = campaign.iloc[sel_idx].copy()
log_msg(f"\nSelected {final_result['n_selected']:,} customers matching constraints.")

# Hangi kısıt bağladı?
max_contacts_budget = int(config.campaign.budget / config.campaign.contact_cost)
roi_uplift_threshold = (config.campaign.contact_cost * (1 + min_roi_constraint)) / config.campaign.margin
n_roi_eligible = int((uplift >= roi_uplift_threshold).sum())

binder = []
if final_result['n_selected'] == max_contacts_budget:
    binder.append("BUDGET")
if final_result['n_selected'] == n_roi_eligible:
    binder.append("ROI")
binder = " & ".join(binder) if binder else "NONE"

log_msg(f"Constraint binding: {binder} "
        f"(ROI threshold uplift ≥ {roi_uplift_threshold:.4f}, "
        f"ROI-eligible={n_roi_eligible:,}, "
        f"Budget-eligible={max_contacts_budget:,})")

# Emniyet: bütçe trim (normalde gerekmez; yine de dursun)
max_contacts = int(config.campaign.budget / config.campaign.contact_cost)
if len(target_list) > max_contacts:
    target_list = target_list.sort_values('cate', ascending=False).head(max_contacts).copy()
    log_msg(f"[INFO] Trimmed to budget: kept top {max_contacts:,} by CATE to respect budget.")

# Kampanya metadata
campaign_id = datetime.now().strftime("CPN%Y%m%d_%H%M%S")
target_list['campaign_id'] = campaign_id
target_list['strategy_name'] = "Optimized (ROI & Budget)"
target_list['optimizer'] = method_label
target_list['constraint'] = constraint_line

# Skor ve finans kolonları
target_list['expected_incremental_revenue'] = target_list['cate'] * config.campaign.margin
target_list['expected_incremental_profit_per_customer'] = (
    target_list['expected_incremental_revenue'] - config.campaign.contact_cost
)
target_list['expected_roi_pct_per_customer'] = (
    target_list['expected_incremental_profit_per_customer'] / config.campaign.contact_cost
).replace([np.inf, -np.inf], np.nan)

target_list = target_list.sort_values('cate', ascending=False)

# Profit–Budget frontier (karar desteği)
profit_per_cust = uplift * config.campaign.margin - config.campaign.contact_cost
order = np.argsort(profit_per_cust)[::-1]
cum_uplift = np.cumsum(uplift[order])
contacts = np.arange(1, len(order) + 1)
costs = contacts * config.campaign.contact_cost
revenues = cum_uplift * config.campaign.margin
profits = revenues - costs
frontier = pd.DataFrame({
    'contacts': contacts,
    'cost': costs,
    'revenue': revenues,
    'profit': profits,
    'roi_pct': np.where(costs > 0, profits / costs * 100, np.nan)
})
frontier.to_csv('results/profit_budget_frontier.csv', index=False)
log_msg("Saved: results/profit_budget_frontier.csv (profit vs. budget frontier)")

# ---------------------------------------------------------------------
# A/B TEST SPLIT (Stratified with exact global ratio; per-stratum quotas)
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("A/B TEST CONFIGURATION")
log_msg("=" * 70)

seed = getattr(config.model, 'random_state', getattr(config.campaign, 'random_state', 42))
rng = np.random.RandomState(int(seed))
control_ratio = float(np.clip(getattr(config.campaign, 'test_split_ratio', 0.2), 0.01, 0.99))

# 1) CATE'yi 10 desile böl
target_list['cate_decile'] = pd.qcut(
    target_list['cate'], 10, labels=False, duplicates='drop'
)

# 2) Tabaka başına Hamilton (largest remainder) ile kontrol kotası
strata = []
total_n = len(target_list)
target_control = int(round(total_n * control_ratio))

for key, grp in target_list.groupby(['rfm_segment', 'cate_decile'], dropna=False):
    n = len(grp)
    if n == 0:
        continue
    if n == 1:
        base = 0
        remainder = 0.0
        min_ctrl, max_ctrl = 0, 0
    else:
        raw = n * control_ratio
        base = int(np.floor(raw))
        remainder = float(raw - base)
        # n>=2 ise her iki grup da en az 1 olsun
        min_ctrl, max_ctrl = 1, n - 1

    strata.append({
        'key': key,
        'index': grp.index.values,   # numpy array
        'n': n,
        'base': base,
        'remainder': remainder,
        'min_ctrl': min_ctrl,
        'max_ctrl': max_ctrl,
        'assigned': None,
    })

# 3) İlk atama: base'i [min,max] içine kırp
for s in strata:
    s['assigned'] = int(np.clip(s['base'], s['min_ctrl'], s['max_ctrl']))

sum_assigned = int(np.sum([s['assigned'] for s in strata]))
delta = int(target_control - sum_assigned)

# 4) Toplamı tam hedefe eşitle (largest remainder) — kapasiteye saygı
if delta > 0:
    candidates = [i for i, s in enumerate(strata) if s['assigned'] < s['max_ctrl']]
    order = sorted(candidates, key=lambda i: (strata[i]['remainder'], strata[i]['n']), reverse=True)
    for i in order:
        if delta == 0:
            break
        room = strata[i]['max_ctrl'] - strata[i]['assigned']
        if room <= 0:
            continue
        strata[i]['assigned'] += 1
        delta -= 1
elif delta < 0:
    candidates = [i for i, s in enumerate(strata) if s['assigned'] > s['min_ctrl']]
    order = sorted(candidates, key=lambda i: (strata[i]['remainder'], strata[i]['n']))  # küçük kalandan başla
    for i in order:
        if delta == 0:
            break
        room = strata[i]['assigned'] - strata[i]['min_ctrl']
        if room <= 0:
            continue
        strata[i]['assigned'] -= 1
        delta += 1

# 5) Atama: her tabakada "assigned" kadar controlü RANDOM seç
target_list['ab_group'] = 'treatment'
for s in strata:
    n = s['n']
    if n == 0 or s['assigned'] == 0:
        continue
    if n == 1:
        continue  # tek kişi varsa max_ctrl=0 olduğundan control seçilmez
    ctrl_size = int(s['assigned'])
    ctrl_idx = rng.choice(s['index'], size=ctrl_size, replace=False)
    target_list.loc[ctrl_idx, 'ab_group'] = 'control'

# 6) Final sayılar ve log
n_control = int((target_list['ab_group'] == 'control').sum())
n_treatment = int((target_list['ab_group'] == 'treatment').sum())

log_msg(f"\nA/B Test Split:")
log_msg(f"  Treatment group: {n_treatment:,} customers ({(n_treatment/len(target_list))*100:.1f}%)")
log_msg(f"  Control group: {n_control:,} customers ({(n_control/len(target_list))*100:.1f}%)")
assert n_control + n_treatment == len(target_list)

# ---------------------------------------------------------------------
# POWER CALCULATION
# ---------------------------------------------------------------------
baseline = float(np.clip(target_list['p_control'].mean(), 0, 1)) if 'p_control' in target_list.columns else 0.0
effect = float(target_list['cate'].mean())
expected_treatment = float(np.clip(baseline + effect, 0, 1))

log_msg(f"\nExpected Effects:")
log_msg(f"  Baseline (control): {baseline*100:.2f}%")
log_msg(f"  Treatment effect: {effect*100:+.2f}pp")
log_msg(f"  Treatment rate: {expected_treatment*100:.2f}%")

alpha = float(getattr(config.campaign, 'alpha', 0.05))
power_target = float(getattr(config.campaign, 'power', 0.8))
beta = 1 - power_target

z_alpha = stats.norm.ppf(1 - alpha/2)
z_beta = stats.norm.ppf(1 - beta)

pooled_p = np.clip((baseline + expected_treatment) / 2, 1e-6, 1 - 1e-6)

if abs(effect) < 1e-6:
    n_per_group_required = float('inf')
else:
    n_per_group_required = ((z_alpha + z_beta) ** 2 * 2 * pooled_p * (1 - pooled_p)) / (effect ** 2)

log_msg(f"\nSample Size:")
if np.isinf(n_per_group_required):
    log_msg("  Required per group: ∞ (effect≈0; test anlamlı değil)")
else:
    log_msg(f"  Required per group: {int(np.ceil(n_per_group_required)):,}")
log_msg(f"  Actual treatment: {n_treatment:,}")
log_msg(f"  Actual control: {n_control:,}")
power_sufficient = (not np.isinf(n_per_group_required)) and (n_control >= n_per_group_required)
log_msg(f"  Status: {'✅ SUFFICIENT POWER' if power_sufficient else '⚠️ UNDERPOWERED'}")

# --------- METADATA KOLONLARINI EKLE (export için) ----------
try:
    campaign_id  # var mı?
except NameError:
    campaign_id = f"cmp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

strategy_name = "Optimized (ROI & Budget)"
optimizer_label = globals().get("method_label", "src.optimize.optimize_with_constraints")
constraint_text = globals().get(
    "constraint_line",
    f"min_ROI ≥ {float(getattr(config.campaign,'min_roi',0.0))*100:.0f}% AND "
    f"Budget ≤ ${float(getattr(config.campaign,'budget',0.0)):,.0f}"
)

target_list['campaign_id']  = campaign_id
target_list['strategy_name'] = strategy_name
target_list['optimizer']     = optimizer_label
target_list['constraint']    = constraint_text

# ================= QA CHECKS ================
# 1) Tekil müşteri, tek grup
assert target_list['client_id'].is_unique, "client_id tekrar ediyor!"
vc = target_list['ab_group'].value_counts()
assert set(vc.index) <= {"control","treatment"}, "ab_group beklenmeyen değer içeriyor!"

# 2) Control oranı tolerans içinde (±0.2pp)
ctrl_pct = (target_list['ab_group']=="control").mean()*100
assert abs(ctrl_pct - control_ratio*100) <= 0.2, f"Control oranı sapması: {ctrl_pct:.2f}%"

# 3) CATE ve finansal kolonlarda NaN yok
for col in ["cate","expected_incremental_revenue","expected_incremental_profit_per_customer"]:
    assert target_list[col].notna().all(), f"{col} içinde NaN var!"

# 4) A/B tabakaları dengeli: n>=2 ise her iki grup da var (>=1)
for k, grp in target_list.groupby(["rfm_segment","cate_decile"], dropna=False):
    if len(grp) <= 1:
        continue
    gvc = grp['ab_group'].value_counts()
    assert ("control" in gvc) and ("treatment" in gvc), f"Tabaka {k} tek gruba düşmüş (n={len(grp)})!"

# ---------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("SAVING OUTPUTS")
log_msg("=" * 70)

# 1) Tüm hedef liste (A/B dahil)
target_list.to_csv('results/target_customers_with_ab.csv', index=False)
log_msg(f"Saved: results/target_customers_with_ab.csv ({len(target_list):,} rows)")

# 2) Treatment list (uygulama)
treatment_list = target_list[target_list['ab_group'] == 'treatment'][[
    'client_id', 'cate', 'expected_incremental_revenue',
    'expected_incremental_profit_per_customer', 'expected_roi_pct_per_customer',
    'rfm_segment', 'monetary_capped', 'aov', 'frequency', 'recency',
    'campaign_id', 'strategy_name', 'optimizer', 'constraint'
]]
treatment_list.to_csv('exports/campaign_treatment_list.csv', index=False)
log_msg(f"Saved: exports/campaign_treatment_list.csv ({len(treatment_list):,} rows)")

# 3) Control list (holdout)
control_list = target_list[target_list['ab_group'] == 'control'][[
    'client_id', 'cate', 'expected_incremental_revenue',
    'expected_incremental_profit_per_customer',
    'rfm_segment', 'monetary_capped',
    'campaign_id', 'strategy_name', 'optimizer', 'constraint'
]]
control_list.to_csv('exports/campaign_control_list.csv', index=False)
log_msg(f"Saved: exports/campaign_control_list.csv ({len(control_list):,} rows)")

# 4) Campaign summary (pickle) — final_result metriklerinden
summary_data = {
    'strategy': "Optimized (ROI & Budget)",
    'n_total': int(len(target_list)),
    'n_treatment': int(n_treatment),
    'n_control': int(n_control),
    'total_cost': float(final_result['total_cost']),
    'expected_revenue': float(final_result['expected_revenue']),
    'expected_profit': float(final_result['expected_profit']),
    'roi_pct': float(final_result['roi_pct']),
    'avg_cate': float(target_list['cate'].mean()),
    'avg_clv': float(target_list['monetary_capped'].mean()),
    'baseline_conversion': float(baseline),
    'expected_uplift': float(effect),
    'power_sufficient': bool(power_sufficient),
    'control_ratio': float(control_ratio),
    'optimizer': method_label,
    'constraint': constraint_line,
    'campaign_id': campaign_id,
}
with open('results/campaign_summary.pkl', 'wb') as f:
    pickle.dump(summary_data, f)
log_msg("Saved: results/campaign_summary.pkl")

# ---------------------------------------------------------------------
# EXECUTIVE SUMMARY — final_result metrikleriyle
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("EXECUTIVE SUMMARY")
log_msg("=" * 70)

summary_text = f"""
CAMPAIGN PLAN
{'=' * 70}

RECOMMENDED STRATEGY: Optimized (ROI & Budget)

TARGET AUDIENCE:
  • Total Size: {len(target_list):,} customers
  • Treatment Group: {n_treatment:,} customers ({(n_treatment/len(target_list))*100:.0f}%)
  • Control Group: {n_control:,} customers ({(n_control/len(target_list))*100:.0f}%)
  • Average CATE: {target_list['cate'].mean():+.2%}
  • Average CLV: ${target_list['monetary_capped'].mean():,.0f}

FINANCIAL PROJECTIONS:
  • Campaign Cost: ${final_result['total_cost']:,.0f}
  • Expected Incremental Revenue: ${final_result['expected_revenue']:,.0f}
  • Net Profit: ${final_result['expected_profit']:,.0f}
  • ROI: {final_result['roi_pct']:.1f}%
  • Cost per Contact: ${config.campaign.contact_cost:.2f}

STATISTICAL DESIGN:
  • Baseline Conversion: {baseline*100:.2f}%
  • Expected Uplift: {effect*100:+.2f}pp
  • Statistical Power: {power_target*100:.0f}%
  • Significance Level: {alpha*100:.0f}%
  • Power Status: {'✅ Sufficient' if power_sufficient else '⚠️ Underpowered'}

OPTIMIZATION METHOD:
  • Algorithm: {method_label}
  • Constraint: {constraint_line}
  • Selection: {final_result['n_selected']:,} customers selected
  • Average Predicted Uplift: {final_result['avg_uplift']:+.4f}

KEY DELIVERABLES:
  ✅ exports/campaign_treatment_list.csv - Send campaign to these customers
  ✅ exports/campaign_control_list.csv - Holdout group (DO NOT CONTACT)
  ✅ results/target_customers_with_ab.csv - Full list with assignments
  ✅ results/campaign_strategies_comparison.csv - Strategy comparison
  ✅ results/profit_budget_frontier.csv - Profit vs. Budget frontier

NEXT STEPS:
  1. Review treatment list (exports/campaign_treatment_list.csv)
  2. Set up campaign in marketing platform
  3. Ensure control group is properly held out
  4. Monitor campaign performance in real-time
  5. Post-campaign: Compare actual vs predicted uplift
"""
log_msg(summary_text)

# ---------------------------------------------------------------------
# SEGMENTATION INSIGHTS
# ---------------------------------------------------------------------
log_msg("\n" + "=" * 70)
log_msg("SEGMENTATION INSIGHTS")
log_msg("=" * 70)

segment_col = 'rfm_tier' if 'rfm_tier' in target_list.columns else 'rfm_segment'
rfm_analysis = (target_list
    .groupby(segment_col)
    .agg(
        n_customers=('client_id', 'count'),
        avg_cate=('cate', 'mean'),
        avg_revenue=('expected_incremental_revenue', 'mean'),
        avg_profit=('expected_incremental_profit_per_customer', 'mean'),
        avg_roi=('expected_roi_pct_per_customer', 'mean'),
    )
    .round(4)
    .sort_values('avg_profit', ascending=False)
)

log_msg("\nTop RFM Segments:")
for segment, row in rfm_analysis.head(10).iterrows():
    log_msg(f"  {segment}: n={int(row['n_customers']):,}, "
            f"CATE={row['avg_cate']:+.4f}, "
            f"Profit/px=${row['avg_profit']:+.2f}, "
            f"ROI/px={row['avg_roi']*100:+.1f}%")

rfm_analysis.to_csv('results/rfm_segment_analysis.csv')
log_msg("\nSaved: results/rfm_segment_analysis.csv")

log_msg("\n" + "=" * 70)
log_msg("CAMPAIGN PLANNING COMPLETE!")
log_msg("=" * 70)

log_msg(f"""
✅ SUCCESS!

Key Files Generated:
  • exports/campaign_treatment_list.csv ({len(treatment_list):,} customers)
  • exports/campaign_control_list.csv ({len(control_list):,} customers)
  • results/campaign_summary.pkl (metrics)
  • results/campaign_strategies_comparison.csv (all strategies)
  • results/profit_budget_frontier.csv (frontier)

Expected Results:
  • Profit: ${final_result['expected_profit']:,.0f}
  • ROI: {final_result['roi_pct']:.1f}%
  
Ready to launch! 🚀
""")

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
def main():
    """Main function for script execution"""
    return {
        'target_list': target_list,
        'treatment_list': treatment_list,
        'control_list': control_list,
        'comparison': comparison_df,
        'summary': summary_data,
        'optimization_result': final_result
    }

if __name__ == '__main__':
    _ = main()
