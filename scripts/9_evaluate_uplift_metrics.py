"""
X5 RetailHero - Uplift Metrics Evaluation (Optimized)
Lightweight, fast evaluation without heavy computation

Metrics:
- Qini AUC: Cumulative gain ranking metric
- Uplift@k: Top k% customer effect
- ATE: Average treatment effect with CI
- Basic treatment balance check
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

Path('results').mkdir(exist_ok=True)
Path('plots').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# ======================== SIMPLE LOGGER ========================

def log_msg(msg, log_file='logs/02_evaluate_metrics.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Initialize log
with open('logs/02_evaluate_metrics.log', 'w') as f:
    f.write(f"Started: {datetime.now()}\n\n")

# ======================== FAST METRICS ========================

def qini_auc_fast(y_true, cate, treatment):
    """Fast Qini AUC calculation"""
    idx_sorted = np.argsort(cate)[::-1]
    y_s = y_true[idx_sorted]
    t_s = treatment[idx_sorted]
    
    n = len(y_true)
    nt = (t_s == 1).sum()
    nc = (t_s == 0).sum()
    
    if nt == 0 or nc == 0:
        return 0.0
    
    # Cumulative gains
    t_idx = np.where(t_s == 1)[0]
    c_idx = np.where(t_s == 0)[0]
    
    y_t = y_s[t_s == 1]
    y_c = y_s[t_s == 0]
    
    cum_t = np.cumsum(y_t)
    cum_c = np.cumsum(y_c)
    
    # Qini at each position
    rate_t = cum_t / np.arange(1, len(y_t) + 1)
    rate_c = cum_c / np.arange(1, len(y_c) + 1)
    
    qini = np.trapz(rate_t) / n - np.trapz(rate_c) / n
    
    return np.clip(qini, -1, 1)


def uplift_at_k_fast(y_true, cate, treatment, k_list=[10, 20, 30, 50]):
    """Fast Uplift@k calculation"""
    idx = np.argsort(cate)[::-1]
    y_s = y_true[idx]
    t_s = treatment[idx]
    
    n = len(y_true)
    results = {}
    
    for k in k_list:
        n_k = int(n * k / 100)
        
        y_k = y_s[:n_k]
        t_k = t_s[:n_k]
        
        mask_t = t_k == 1
        mask_c = t_k == 0
        
        if mask_t.sum() > 0 and mask_c.sum() > 0:
            uplift = (y_k[mask_t].mean() - y_k[mask_c].mean()) * 100
        else:
            uplift = 0.0
        
        results[f'uplift_at_{k}'] = uplift
    
    return results


def ate_ci(y_true, treatment, conf=0.95):
    """ATE with confidence interval"""
    y_t = y_true[treatment == 1]
    y_c = y_true[treatment == 0]
    
    ate = y_t.mean() - y_c.mean()
    
    se_t = y_t.std() / np.sqrt(len(y_t))
    se_c = y_c.std() / np.sqrt(len(y_c))
    se = np.sqrt(se_t**2 + se_c**2)
    
    z = stats.norm.ppf((1 + conf) / 2)
    ci_l = ate - z * se
    ci_u = ate + z * se
    
    return {
        'ate': ate * 100,
        'ci_lower': ci_l * 100,
        'ci_upper': ci_u * 100
    }


def treatment_balance_fast(X_test, treatment):
    """Fast SMD calculation (sample every 10th feature if too many)"""
    X_t = X_test[treatment == 1]
    X_c = X_test[treatment == 0]
    
    n_feat = X_test.shape[1]
    step = max(1, n_feat // 25)  # Max 25 features to check
    
    smd_list = []
    
    for i in range(0, n_feat, step):
        x_t = X_t[:, i]
        x_c = X_c[:, i]
        
        if x_t.std() + x_c.std() > 0:
            smd = abs(x_t.mean() - x_c.mean()) / np.sqrt((x_t.var() + x_c.var()) / 2 + 1e-10)
        else:
            smd = 0.0
        
        smd_list.append(smd)
    
    avg_smd = np.mean(smd_list) if smd_list else 0.0
    
    return {
        'avg_smd': avg_smd,
        'status': 'Good' if avg_smd < 0.1 else ('OK' if avg_smd < 0.2 else 'Poor')
    }


# ======================== SIMPLE PLOTS ========================

def plot_qini(y_true, cate, treatment):
    """Fast Qini curve plot"""
    idx = np.argsort(cate)[::-1]
    y_s = y_true[idx]
    t_s = treatment[idx]
    
    n = len(y_true)
    step = max(1, n // 100)
    
    qini_vals = []
    pcts = []
    
    for i in range(0, n, step):
        mask_t = t_s[:i+1] == 1
        mask_c = t_s[:i+1] == 0
        
        if mask_t.sum() > 0 and mask_c.sum() > 0:
            uplift = (y_s[:i+1][mask_t].mean() - y_s[:i+1][mask_c].mean()) * 100
            qini_vals.append(uplift)
            pcts.append((i + 1) / n * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pcts, qini_vals, linewidth=2.5, color='#2E86AB')
    ax.fill_between(pcts, qini_vals, alpha=0.3, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Customers Targeted (%)', fontsize=11)
    ax.set_ylabel('Uplift (%)', fontsize=11)
    ax.set_title('Qini Cumulative Gain', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/01_qini_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_msg("Saved: plots/01_qini_curve.png")


def plot_cate_dist(cate, treatment):
    """CATE distribution plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(cate[treatment == 0], bins=40, alpha=0.6, label='Control', color='#A23B72')
    ax.hist(cate[treatment == 1], bins=40, alpha=0.6, label='Treatment', color='#F18F01')
    
    ax.set_xlabel('CATE', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('CATE Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/02_cate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_msg("Saved: plots/02_cate_distribution.png")


def plot_uplift_k(uplift_dict):
    """Uplift@k bar chart"""
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
    log_msg("UPLIFT METRICS EVALUATION (Fast)")
    log_msg("=" * 70)
    
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
    
    # Metrics
    log_msg("\n" + "=" * 70)
    log_msg("METRICS")
    log_msg("=" * 70)
    
    # Qini AUC
    log_msg("\n[1] Qini AUC")
    qini = qini_auc_fast(y_true, cate, treatment)
    log_msg(f"Score: {qini:.4f}")
    log_msg("Status: EXCELLENT" if qini > 0.15 else ("GOOD" if qini > 0.05 else "FAIR"))
    
    # Uplift@k
    log_msg("\n[2] Uplift@k")
    u_k = uplift_at_k_fast(y_true, cate, treatment)
    for k, v in sorted(u_k.items()):
        log_msg(f"{k}: {v:+.2f}%")
    
    # ATE
    log_msg("\n[3] Average Treatment Effect")
    ate_res = ate_ci(y_true, treatment)
    log_msg(f"ATE: {ate_res['ate']:+.2f}%")
    log_msg(f"95% CI: [{ate_res['ci_lower']:+.2f}%, {ate_res['ci_upper']:+.2f}%]")
    
    # CATE stats
    log_msg("\n[4] CATE Statistics")
    log_msg(f"Mean: {cate.mean():+.4f} | Std: {cate.std():.4f}")
    log_msg(f"Range: [{cate.min():+.4f}, {cate.max():+.4f}]")
    
    # Treatment balance
    log_msg("\n[5] Treatment Balance")
    excl = {'client_id', 'target', 'treatment', 'rfm_segment', 'rfm_score', 
            'segment', 'r_score', 'f_score', 'm_score'}
    feat_cols = [c for c in df.columns if c not in excl and df[c].dtype in ['int64', 'float64']]
    
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, 
                                   random_state=42, stratify=df['treatment'])
    
    X_test = df.iloc[test_idx][feat_cols].values
    T_test = df.iloc[test_idx]['treatment'].values
    
    bal = treatment_balance_fast(X_test, T_test)
    log_msg(f"Average SMD: {bal['avg_smd']:.4f} ({bal['status']})")
    
    # Save metrics
    log_msg("\n" + "=" * 70)
    log_msg("SAVING RESULTS")
    log_msg("=" * 70)
    
    metrics_df = pd.DataFrame({
        'metric': ['Qini AUC', 'ATE (%)', 'ATE CI Lower (%)', 'ATE CI Upper (%)'] + list(u_k.keys()),
        'value': [qini, ate_res['ate'], ate_res['ci_lower'], ate_res['ci_upper']] + list(u_k.values())
    })
    metrics_df.to_csv('results/uplift_metrics.csv', index=False)
    log_msg("Saved: results/uplift_metrics.csv")
    
    # Generate plots
    log_msg("\nGenerating plots...")
    plot_qini(y_true, cate, treatment)
    plot_cate_dist(cate, treatment)
    plot_uplift_k(u_k)
    
    log_msg("\n" + "=" * 70)
    log_msg("COMPLETE!")
    log_msg("=" * 70)
    
    return metrics_df

if __name__ == '__main__':
    metrics_df = main()