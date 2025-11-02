"""
Model Comparison Script (FIXED)
Compare T-Learner, S-Learner, and X-Learner performance on FULL dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics import (
    qini_auc_score,
    uplift_at_k_multiple,
    average_treatment_effect
)

def calculate_metrics(pred_file, data_file='data/x5_rfm_processed.pkl'):
    """
    Calculate metrics for a model predictions file
    
    IMPORTANT: Uses FULL dataset, not test split!
    """
    # Load predictions
    pred_df = pd.read_csv(pred_file)
    
    # Load actual data
    import pickle
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    df = data['data']
    
    # Merge
    merged = df[['client_id', 'treatment', 'target']].merge(pred_df, on='client_id')
    
    y_true = merged['target'].values
    cate = merged['cate'].values
    treatment = merged['treatment'].values
    
    print(f"  Samples: {len(merged):,}")
    print(f"  Treatment: {(treatment==1).sum():,} | Control: {(treatment==0).sum():,}")
    
    # Calculate metrics - DIRECTLY on full data
    qini = qini_auc_score(y_true, cate, treatment)
    uplift_k = uplift_at_k_multiple(y_true, cate, treatment, k_list=[0.1, 0.2, 0.3, 0.5])
    ate = average_treatment_effect(y_true, treatment)
    
    return {
        'qini_auc': qini,
        'uplift_at_10': uplift_k['uplift_at_10'],
        'uplift_at_20': uplift_k['uplift_at_20'],
        'uplift_at_30': uplift_k['uplift_at_30'],
        'uplift_at_50': uplift_k['uplift_at_50'],
        'ate': ate['ate'],
        'samples': len(merged)
    }

def compare_models():
    """Compare all available models on FULL dataset"""
    print("=" * 70)
    print("MODEL COMPARISON (Full Dataset)")
    print("=" * 70)
    
    models = {
        'T-Learner': 'results/tlearner_predictions.csv',
        'S-Learner': 'results/slearner_predictions.csv',
        'X-Learner': 'results/xlearner_predictions.csv',
        'R-Learner': 'results/rlearner_predictions.csv'  # If exists
    }
    
    results = {}
    
    for name, file in models.items():
        if os.path.exists(file):
            print(f"\nEvaluating {name}...")
            try:
                metrics = calculate_metrics(file)
                results[name] = metrics
                print(f"✓ {name}: Qini AUC = {metrics['qini_auc']:.6f}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
        else:
            print(f"✗ {name}: File not found")
    
    if not results:
        print("\n❌ No model results found!")
        return
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('qini_auc', ascending=False)
    
    print("\nFull Results:")
    print(comparison_df.to_string())
    
    # Save results
    comparison_df.to_csv('results/model_comparison.csv')
    print(f"\n✓ Saved: results/model_comparison.csv")
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("DETAILED RANKING")
    print("=" * 70)
    
    for i, (model, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"\n{i}. {model}:")
        print(f"   Qini AUC:    {row['qini_auc']:.6f}")
        print(f"   Uplift@10:   {row['uplift_at_10']:.2f}%")
        print(f"   Uplift@20:   {row['uplift_at_20']:.2f}%")
        print(f"   Uplift@30:   {row['uplift_at_30']:.2f}%")
        print(f"   Uplift@50:   {row['uplift_at_50']:.2f}%")
        print(f"   ATE:         {row['ate']*100:.2f}%")
    
    # Find best model
    best_model = comparison_df.index[0]
    best_qini = comparison_df.loc[best_model, 'qini_auc']
    
    print(f"\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   Qini AUC: {best_qini:.6f}")
    print(f"   Uplift@10: {comparison_df.loc[best_model, 'uplift_at_10']:.2f}%")
    print("=" * 70)
    
    # Performance gaps
    if len(comparison_df) > 1:
        print("\n📊 Performance Gaps:")
        best_qini = comparison_df.iloc[0]['qini_auc']
        
        for i, (model, row) in enumerate(comparison_df.iterrows(), 1):
            if i == 1:
                continue
            gap = ((best_qini - row['qini_auc']) / row['qini_auc']) * 100
            print(f"   {model}: {gap:.1f}% worse than {comparison_df.index[0]}")
    
    return comparison_df

if __name__ == '__main__':
    comparison_df = compare_models()