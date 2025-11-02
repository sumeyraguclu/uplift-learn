"""
X5 RetailHero - CATE Calibration (Refactored)
Uses centralized src.calibration module

Calibrates raw CATE predictions to improve reliability
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# Import from src modules
from src.calibration import CATECalibrator, calibrate_cate
from src.config import get_config

warnings.filterwarnings('ignore')

# Get config
config = get_config()
config.ensure_dirs()

def log_msg(msg, log_file='logs/12_calibration.log'):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Initialize log
with open('logs/12_calibration.log', 'w') as f:
    f.write(f"CATE Calibration (Refactored)\nStarted: {datetime.now()}\n\n")

# ======================== LOAD DATA ========================

log_msg("=" * 70)
log_msg("CATE CALIBRATION")
log_msg("=" * 70)
log_msg("Using src.calibration module")

log_msg("\nLoading data...")

# Load predictions
pred_df = pd.read_csv(config.paths.predictions)

# Load actual outcomes
with open(config.paths.rfm_data, 'rb') as f:
    data = pickle.load(f)
df = data['data']

# Merge
merged = df[['client_id', 'treatment', 'target', 'rfm_segment', 'monetary_capped']].merge(
    pred_df, on='client_id'
)

log_msg(f"Total customers: {len(merged):,}")
log_msg(f"Actual conversion rate: {merged['target'].mean()*100:.2f}%")
log_msg(f"Mean raw CATE: {merged['cate'].mean():+.4f}")

# ======================== CALIBRATION ========================

log_msg("\n" + "=" * 70)
log_msg("CALIBRATING CATE (using src.calibration.calibrate_cate)")
log_msg("=" * 70)

# Calibrate using convenience function
calibrated_df, calibrator = calibrate_cate(
    predictions_df=pred_df,
    outcomes_df=df,
    save_calibrator=True,
    calibrator_path=config.paths.calibrator,
    verbose=True
)

# ======================== STATISTICS ========================

log_msg("\n" + "=" * 70)
log_msg("CALIBRATED CATE STATISTICS")
log_msg("=" * 70)

log_msg(f"\nRaw CATE:")
log_msg(f"  Mean: {calibrated_df['cate'].mean():+.4f}")
log_msg(f"  Std: {calibrated_df['cate'].std():.4f}")
log_msg(f"  Range: [{calibrated_df['cate'].min():+.4f}, {calibrated_df['cate'].max():+.4f}]")

log_msg(f"\nCalibrated CATE:")
log_msg(f"  Mean: {calibrated_df['cate_calibrated'].mean():+.4f}")
log_msg(f"  Std: {calibrated_df['cate_calibrated'].std():.4f}")
log_msg(f"  Range: [{calibrated_df['cate_calibrated'].min():+.4f}, {calibrated_df['cate_calibrated'].max():+.4f}]")

# Get calibration metrics
metrics = calibrator.get_metrics()
log_msg(f"\nCalibration Quality:")
log_msg(f"  Treatment MAE: {metrics['mae_before_treatment']:.4f} → {metrics['mae_after_treatment']:.4f}")
log_msg(f"  Control MAE: {metrics['mae_before_control']:.4f} → {metrics['mae_after_control']:.4f}")
log_msg(f"  Improvement: {metrics['improvement_treatment']*100:.1f}% / {metrics['improvement_control']*100:.1f}%")

# ======================== SEGMENT ANALYSIS ========================

log_msg("\n" + "=" * 70)
log_msg("SEGMENT-LEVEL ANALYSIS")
log_msg("=" * 70)

# Merge with segment info
calibrated_with_segments = calibrated_df.merge(
    df[['client_id', 'rfm_segment', 'target']], on='client_id'
)

# Analyze by segment
segment_comparison = calibrated_with_segments.groupby('rfm_segment').agg({
    'cate': 'mean',
    'cate_calibrated': 'mean',
    'client_id': 'count',
    'target': 'mean'
}).round(4)

segment_comparison.columns = ['raw_cate_mean', 'calibrated_cate_mean', 'n_customers', 'conversion_rate']
segment_comparison['cate_change'] = segment_comparison['calibrated_cate_mean'] - segment_comparison['raw_cate_mean']
segment_comparison = segment_comparison.sort_values('calibrated_cate_mean', ascending=False)

log_msg("\nTop 10 Segments (by calibrated CATE):")
for seg, row in segment_comparison.head(10).iterrows():
    log_msg(f"  {seg}: raw={row['raw_cate_mean']:+.4f}, "
            f"calibrated={row['calibrated_cate_mean']:+.4f}, "
            f"change={row['cate_change']:+.4f}, "
            f"n={int(row['n_customers']):,}")

# ======================== SAVE OUTPUTS ========================

log_msg("\n" + "=" * 70)
log_msg("SAVING RESULTS")
log_msg("=" * 70)

# Save calibrated predictions
calibrated_df.to_csv('results/calibrated_cate.csv', index=False)
log_msg(f"Saved: results/calibrated_cate.csv")

# Save final CATE (use calibrated version)
final_cate_df = calibrated_df[['client_id', 'p_control', 'p_treatment', 
                                'p_control_cal', 'p_treatment_cal',
                                'cate', 'cate_calibrated']].copy()
final_cate_df['cate_final'] = final_cate_df['cate_calibrated']  # Use calibrated
final_cate_df.to_csv('results/final_cate.csv', index=False)
log_msg(f"Saved: results/final_cate.csv (with calibrated CATE)")

# Save segment analysis
segment_comparison.to_csv('results/segment_calibration_analysis.csv')
log_msg(f"Saved: results/segment_calibration_analysis.csv")

# ======================== VISUALIZATION ========================

log_msg("\n" + "=" * 70)
log_msg("GENERATING PLOTS")
log_msg("=" * 70)

# Plot 1: Calibration curves
log_msg("Generating calibration curves...")
calibrator.plot_calibration(
    p_treatment=merged['p_treatment'].values,
    p_control=merged['p_control'].values,
    y_true=merged['target'].values,
    treatment=merged['treatment'].values,
    save_path='plots/12_calibration_curves.png'
)
log_msg("Saved: plots/12_calibration_curves.png")

# Plot 2: CATE comparison (raw vs calibrated)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Raw CATE
ax1.hist(calibrated_df['cate'], bins=100, alpha=0.7, color='#2E86AB', edgecolor='black')
ax1.axvline(calibrated_df['cate'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {calibrated_df["cate"].mean():.4f}')
ax1.set_xlabel('Raw CATE', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Raw CATE Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Calibrated CATE
ax2.hist(calibrated_df['cate_calibrated'], bins=100, alpha=0.7, color='#06A77D', edgecolor='black')
ax2.axvline(calibrated_df['cate_calibrated'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {calibrated_df["cate_calibrated"].mean():.4f}')
ax2.set_xlabel('Calibrated CATE', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Calibrated CATE Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/12_cate_comparison.png', dpi=config.plotting.dpi, bbox_inches='tight')
log_msg("Saved: plots/12_cate_comparison.png")
plt.close()

# Plot 3: Raw vs Calibrated scatter
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(calibrated_df['cate'], calibrated_df['cate_calibrated'], 
          alpha=0.1, s=10, color='#2E86AB')
ax.plot([calibrated_df['cate'].min(), calibrated_df['cate'].max()],
       [calibrated_df['cate'].min(), calibrated_df['cate'].max()],
       'r--', linewidth=2, label='No Change Line')
ax.set_xlabel('Raw CATE', fontsize=12)
ax.set_ylabel('Calibrated CATE', fontsize=12)
ax.set_title('Raw vs Calibrated CATE', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/12_raw_vs_calibrated.png', dpi=config.plotting.dpi, bbox_inches='tight')
log_msg("Saved: plots/12_raw_vs_calibrated.png")
plt.close()

# ======================== SUMMARY ========================

log_msg("\n" + "=" * 70)
log_msg("CALIBRATION COMPLETE")
log_msg("=" * 70)

log_msg(f"""
✓ SUMMARY
├─ Total customers: {len(calibrated_df):,}
├─ Raw CATE mean: {calibrated_df['cate'].mean():+.4f}
├─ Calibrated CATE mean: {calibrated_df['cate_calibrated'].mean():+.4f}
├─ Treatment MAE improvement: {metrics['improvement_treatment']*100:.1f}%
└─ Control MAE improvement: {metrics['improvement_control']*100:.1f}%

✓ OUTPUTS
├─ results/calibrated_cate.csv (all predictions)
├─ results/final_cate.csv (use this for optimization!)
├─ models/calibrator.pkl (reusable calibrator)
├─ results/segment_calibration_analysis.csv
└─ plots/12_* (calibration visualizations)

✓ NEXT STEP
└─ Run campaign planning with calibrated CATE
""")

log_msg("\n" + "=" * 70)


if __name__ == '__main__':
    print("\n✅ CATE calibration complete!")
    print("\nKey improvements:")
    print(f"  • Treatment predictions: {metrics['improvement_treatment']*100:.1f}% better")
    print(f"  • Control predictions: {metrics['improvement_control']*100:.1f}% better")
    print("\nNext: Use results/final_cate.csv for campaign planning")