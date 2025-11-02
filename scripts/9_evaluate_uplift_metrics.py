"""
X5 RetailHero - Uplift Metrics Evaluation (Refactored)
Uses centralized src.metrics module

Metrics:
- Qini AUC: Cumulative gain ranking metric
- Uplift@k: Top k% customer effect
- ATE: Average treatment effect with CI
- Treatment balance check
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings

# Import from src module
from src.metrics import (
    qini_auc_score,
    uplift_at_k_multiple,
    average_treatment_effect,
    treatment_balance_check,
    qini_curve_data,
    evaluate_uplift_model
)

warnings.filterwarnings('ignore')

Path('results').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# ======================== SIMPLE LOGGER ========================

def log_msg(msg, log_file='logs/9_evaluate_metrics.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Initialize log
with open('logs/9_evaluate_metrics.log', 'w') as f:
    f.write(f"Uplift Metrics Evaluation\nStarted: {datetime.now()}\n\n")

# ======================== VISUALIZATION ========================

def plot_qini_curve(y_true, cate, treatment):
    """Qini curve plot using src.metrics"""
    log_msg("Generating Qini curve...")
    
    curve_data = qini_curve_data(y_true, cate, treatment, n_points=100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_data['x'], curve_data['y'], linewidth=2.5, color='#2E86AB', label='Model')
    ax.fill_between(curve_data['x'], curve_data['y'], alpha=0.3, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Customers Targeted (%)', fontsize=11)
    ax.set_ylabel('Uplift (%)', fontsize=11)
    ax.set_title('Qini Cumulative Gain Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/01_qini_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_msg("Saved: plots/01_qini_curve.png")


def plot_cate_distribution(cate, treatment):
    """CATE distribution by group"""
    log_msg("Generating CATE distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(cate[treatment == 0], bins=40, alpha=0.6, label='Control', color='#A23B72')
    ax.hist(cate[treatment == 1], bins=40, alpha=0.6, label='Treatment', color='#F18F01')
    
    ax.axvline(cate.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {cate.mean():.4f}')
    ax.set_xlabel('CATE', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('CATE Distribution by Group', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/02_cate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_msg("Saved: plots/02_cate_distribution.png")


def plot_uplift_at_k(uplift_dict):
    """Uplift@k bar chart"""
    log_msg("Generating Uplift@k plot...")
    
    k_vals = sorted([int(k.split('_')[-1]) for k in uplift_dict.keys()])
    uplifts = [uplift_dict[f'uplift_at_{k}'] for k in k_vals]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = ['#06A77D' if u > 0 else '#D62828' for u in uplifts]
    bars = ax.bar(range(len(k_vals)), uplifts, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f'Top {k}%' for k in k_vals])
    ax.set_ylabel('Uplift (%)', fontsize=11)
    ax.set_title('Uplift by Customer Segment', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(alpha=0.3, axis='y')
    
    for bar, val in zip(bars, uplifts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/03_uplift_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_msg("Saved: plots/03_uplift_at_k.png")


# ======================== MAIN ========================

def main():
    log_msg("=" * 70)
    log_msg("UPLIFT METRICS EVALUATION")
    log_msg("=" * 70)
    log_msg("Using src.metrics module")
    
    # Load data
    log_msg("\nLoading data...")
    pred_df = pd.read_csv('results/tlearner_predictions.csv')
    
    with open('data/x5_rfm_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['data']
    
    merged = df[['client_id', 'treatment', 'target']].merge(pred_df, on='client_id')
    
    y_true = merged['target'].values
    cate = merged['cate'].values
    treatment = merged['treatment'].values
    
    log_msg(f"Samples: {len(merged):,}")
    log_msg(f"Treatment: {(treatment==1).sum():,} | Control: {(treatment==0).sum():,}")
    log_msg(f"Baseline conversion: {y_true.mean()*100:.2f}%")
    
    # Prepare features for balance check
    excl = {'client_id', 'target', 'treatment', 'rfm_segment', 'rfm_score', 
            'segment', 'r_score', 'f_score', 'm_score'}
    feat_cols = [c for c in df.columns if c not in excl and df[c].dtype in ['int64', 'float64']]
    
    _, test_idx = train_test_split(
        np.arange(len(df)), 
        test_size=0.2, 
        random_state=42, 
        stratify=df['treatment']
    )
    
    X_test = df.iloc[test_idx][feat_cols].values
    T_test = df.iloc[test_idx]['treatment'].values
    
    # ============ COMPREHENSIVE EVALUATION ============
    log_msg("\n" + "=" * 70)
    log_msg("COMPREHENSIVE METRICS (using src.metrics.evaluate_uplift_model)")
    log_msg("=" * 70)
    
    # Get test set indices to match X_test and treatment
    y_test = merged.iloc[test_idx]['target'].values
    cate_test = merged.iloc[test_idx]['cate'].values
    treatment_test = merged.iloc[test_idx]['treatment'].values
    
    metrics = evaluate_uplift_model(
        y_true=y_test,
        uplift=cate_test,
        treatment=treatment_test,
        X=X_test,
        k_list=[0.1, 0.2, 0.3, 0.5]
    )
    
    # ============ DISPLAY RESULTS ============
    
    # Qini AUC
    log_msg("\n[1] Qini AUC")
    qini = metrics['qini_auc']
    log_msg(f"Score: {qini:.4f}")
    if qini > 0.15:
        status = "EXCELLENT"
    elif qini > 0.05:
        status = "GOOD"
    else:
        status = "FAIR"
    log_msg(f"Status: {status}")
    
    # Uplift@k
    log_msg("\n[2] Uplift@k")
    for k, v in sorted(metrics['uplift_at_k'].items()):
        log_msg(f"{k}: {v:+.2f}%")
    
    # ATE
    log_msg("\n[3] Average Treatment Effect")
    ate = metrics['ate']
    log_msg(f"ATE: {ate['ate']*100:+.2f}%")
    log_msg(f"95% CI: [{ate['ci_lower']*100:+.2f}%, {ate['ci_upper']*100:+.2f}%]")
    log_msg(f"Std Error: {ate['se']*100:.4f}%")
    
    # CATE stats
    log_msg("\n[4] CATE Statistics")
    log_msg(f"Mean: {cate.mean():+.4f} | Std: {cate.std():.4f}")
    log_msg(f"Range: [{cate.min():+.4f}, {cate.max():+.4f}]")
    log_msg(f"Median: {np.median(cate):+.4f}")
    
    # Treatment balance
    log_msg("\n[5] Treatment Balance (Test Set)")
    balance = metrics['balance']
    log_msg(f"Average SMD: {balance['avg_smd']:.4f}")
    log_msg(f"Maximum SMD: {balance['max_smd']:.4f}")
    log_msg(f"Status: {balance['status']}")
    log_msg(f"Is Balanced: {balance['is_balanced']}")
    
    # ============ SAVE RESULTS ============
    log_msg("\n" + "=" * 70)
    log_msg("SAVING RESULTS")
    log_msg("=" * 70)
    
    # Create metrics DataFrame
    metrics_list = []
    
    # Qini AUC
    metrics_list.append({'metric': 'Qini AUC', 'value': qini})
    
    # ATE
    metrics_list.append({'metric': 'ATE (%)', 'value': ate['ate'] * 100})
    metrics_list.append({'metric': 'ATE CI Lower (%)', 'value': ate['ci_lower'] * 100})
    metrics_list.append({'metric': 'ATE CI Upper (%)', 'value': ate['ci_upper'] * 100})
    
    # Uplift@k
    for k, v in sorted(metrics['uplift_at_k'].items()):
        metrics_list.append({'metric': k, 'value': v})
    
    # Balance
    metrics_list.append({'metric': 'Avg SMD', 'value': balance['avg_smd']})
    metrics_list.append({'metric': 'Max SMD', 'value': balance['max_smd']})
    
    # CATE stats
    metrics_list.append({'metric': 'CATE Mean', 'value': cate.mean()})
    metrics_list.append({'metric': 'CATE Std', 'value': cate.std()})
    metrics_list.append({'metric': 'CATE Min', 'value': cate.min()})
    metrics_list.append({'metric': 'CATE Max', 'value': cate.max()})
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('results/uplift_metrics.csv', index=False)
    log_msg("Saved: results/uplift_metrics.csv")
    
    # ============ GENERATE PLOTS ============
    log_msg("\n" + "=" * 70)
    log_msg("GENERATING PLOTS")
    log_msg("=" * 70)
    
    plot_qini_curve(y_true, cate, treatment)
    plot_cate_distribution(cate, treatment)
    plot_uplift_at_k(metrics['uplift_at_k'])
    
    # ============ SUMMARY ============
    log_msg("\n" + "=" * 70)
    log_msg("EVALUATION COMPLETE!")
    log_msg("=" * 70)
    
    log_msg(f"""
✓ SUMMARY
├─ Qini AUC: {qini:.4f} ({status})
├─ ATE: {ate['ate']*100:+.2f}%
├─ Treatment Balance: {balance['status']}
├─ Mean CATE: {cate.mean():+.4f}
└─ Samples: {len(merged):,}

✓ OUTPUTS
├─ results/uplift_metrics.csv
├─ plots/01_qini_curve.png
├─ plots/02_cate_distribution.png
└─ plots/03_uplift_at_k.png

✓ NEXT STEPS
└─ Run campaign planning: python scripts/10_campaign_planning.py
""")
    
    return metrics_df, metrics


if __name__ == '__main__':
    metrics_df, metrics = main()