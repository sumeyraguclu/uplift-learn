"""
Comprehensive Model Comparison & Validation
Çoklu metrik, cross-validation ve istatistiksel testlerle model karşılaştırması
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.metrics import (
    qini_auc_score,
    uplift_at_k,
    average_treatment_effect,
    evaluate_uplift_model,
    treatment_balance_check
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_FILE = 'data/x5_rfm_processed.pkl'
N_SPLITS = 5  # K-fold CV için
RANDOM_STATE = 42

MODEL_FILES = {
    'T-Learner': 'results/tlearner_predictions.csv',
    'S-Learner': 'results/slearner_predictions.csv',
    'X-Learner': 'results/xlearner_predictions.csv',
}

# R-Learner varsa ekle
if Path('results/rlearner_predictions.csv').exists():
    MODEL_FILES['R-Learner'] = 'results/rlearner_predictions.csv'

# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------

def load_model_predictions(pred_file: str) -> pd.DataFrame:
    """Model prediction dosyasını yükle"""
    return pd.read_csv(pred_file)


def calculate_all_metrics(
    y_true: np.ndarray,
    cate: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray = None
) -> Dict:
    """Tüm metrikleri hesapla"""
    metrics = evaluate_uplift_model(y_true, cate, treatment, X=X)
    
    # Ek metrikler
    cate_mean = cate.mean()
    cate_std = cate.std()
    cate_median = np.median(cate)
    
    # Negatif uplift oranı (sleeping dogs)
    negative_uplift_ratio = (cate < 0).mean()
    
    # Yüksek uplift oranı (top 10%)
    top_10_uplift = np.percentile(cate, 90)
    high_uplift_ratio = (cate > top_10_uplift).mean()
    
    results = {
        'qini_auc': metrics['qini_auc'],
        'uplift_at_10': metrics['uplift_at_k']['uplift_at_10'],
        'uplift_at_20': metrics['uplift_at_k']['uplift_at_20'],
        'uplift_at_30': metrics['uplift_at_k']['uplift_at_30'],
        'uplift_at_50': metrics['uplift_at_k']['uplift_at_50'],
        'ate': metrics['ate']['ate'],
        'ate_ci_lower': metrics['ate']['ci_lower'],
        'ate_ci_upper': metrics['ate']['ci_upper'],
        'cate_mean': cate_mean,
        'cate_std': cate_std,
        'cate_median': cate_median,
        'negative_uplift_ratio': negative_uplift_ratio * 100,
        'high_uplift_ratio': high_uplift_ratio * 100,
    }
    
    if 'balance' in metrics:
        results['avg_smd'] = metrics['balance']['avg_smd']
        results['balance_status'] = metrics['balance']['status']
    
    return results


def cross_validate_model(
    pred_df: pd.DataFrame,
    data_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Cross-validation ile model performansını değerlendir
    """
    # Merge
    merged = data_df[['client_id', 'treatment', 'target']].merge(
        pred_df, on='client_id', how='inner'
    )
    
    if len(merged) == 0:
        print(f"  ⚠️ Warning: No matching rows after merge!")
        return {
            'qini_auc': [0.0] * n_splits,
            'uplift_at_10': [0.0] * n_splits,
            'uplift_at_20': [0.0] * n_splits,
            'uplift_at_30': [0.0] * n_splits
        }
    
    y_true = merged['target'].values
    cate = merged['cate'].values
    treatment = merged['treatment'].values
    
    # KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_results = {
        'qini_auc': [],
        'uplift_at_10': [],
        'uplift_at_20': [],
        'uplift_at_30': []
    }
    
    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(merged)):
            # Validation set
            y_val = y_true[val_idx]
            cate_val = cate[val_idx]
            treatment_val = treatment[val_idx]
            
            # Metrikleri hesapla
            metrics = evaluate_uplift_model(y_val, cate_val, treatment_val)
            
            cv_results['qini_auc'].append(metrics['qini_auc'])
            cv_results['uplift_at_10'].append(metrics['uplift_at_k']['uplift_at_10'])
            cv_results['uplift_at_20'].append(metrics['uplift_at_k']['uplift_at_20'])
            cv_results['uplift_at_30'].append(metrics['uplift_at_k']['uplift_at_30'])
    except Exception as e:
        print(f"  ⚠️ Error in CV: {e}")
        # Fallback: single evaluation
        metrics = evaluate_uplift_model(y_true, cate, treatment)
        cv_results['qini_auc'] = [metrics['qini_auc']] * n_splits
        cv_results['uplift_at_10'] = [metrics['uplift_at_k']['uplift_at_10']] * n_splits
        cv_results['uplift_at_20'] = [metrics['uplift_at_k']['uplift_at_20']] * n_splits
        cv_results['uplift_at_30'] = [metrics['uplift_at_k']['uplift_at_30']] * n_splits
    
    return cv_results


def statistical_significance_test(
    model1_metrics: List[float],
    model2_metrics: List[float],
    metric_name: str = "Metric"
) -> Dict:
    """
    İki model arasında istatistiksel anlamlılık testi (paired t-test)
    """
    if len(model1_metrics) != len(model2_metrics):
        return {
            'test': 'failed',
            'reason': 'Unequal sample sizes'
        }
    
    # Paired t-test
    differences = np.array(model1_metrics) - np.array(model2_metrics)
    t_stat, p_value = stats.ttest_1samp(differences, 0)
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Effect size (Cohen's d)
    cohens_d = mean_diff / (std_diff + 1e-10)
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    significant = p_value < 0.05
    
    return {
        'metric': metric_name,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'cohens_d': cohens_d,
        'effect_size': effect_size
    }


def compare_all_models_comprehensive() -> pd.DataFrame:
    """
    Tüm modelleri kapsamlı şekilde karşılaştır
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON & VALIDATION")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    with open(DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)
    data_df = data_dict['data']
    
    print(f"✓ Data loaded: {len(data_df):,} samples")
    
    # Load models
    print(f"\n🤖 Loading model predictions...")
    model_predictions = {}
    
    for model_name, pred_file in MODEL_FILES.items():
        if Path(pred_file).exists():
            model_predictions[model_name] = load_model_predictions(pred_file)
            print(f"  ✓ {model_name}: {len(model_predictions[model_name]):,} predictions")
        else:
            print(f"  ✗ {model_name}: File not found - {pred_file}")
    
    if not model_predictions:
        print("\n❌ No model predictions found!")
        return None
    
    # -----------------------------------------------------------------
    # 1. FULL DATASET EVALUATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. FULL DATASET EVALUATION")
    print("=" * 80)
    
    full_results = {}
    
    for model_name, pred_df in model_predictions.items():
        print(f"\n📈 Evaluating {model_name}...")
        
        # Merge
        merged = data_df[['client_id', 'treatment', 'target']].merge(
            pred_df, on='client_id', how='inner'
        )
        
        # Extract features for balance check (if available)
        exclude_cols = ['client_id', 'target', 'treatment', 'rfm_segment', 
                        'p_control', 'p_treatment', 'cate', 'cate_pct']
        feature_cols = [c for c in data_df.columns if c not in exclude_cols]
        
        if feature_cols:
            X = merged[feature_cols].fillna(0).values
        else:
            X = None
        
        y_true = merged['target'].values
        cate = merged['cate'].values
        treatment = merged['treatment'].values
        
        # Calculate all metrics
        metrics = calculate_all_metrics(y_true, cate, treatment, X=X)
        full_results[model_name] = metrics
        
        print(f"  Qini AUC:        {metrics['qini_auc']:.4f}")
        print(f"  Uplift@10:       {metrics['uplift_at_10']:.2f}%")
        print(f"  Uplift@20:       {metrics['uplift_at_20']:.2f}%")
        print(f"  ATE:             {metrics['ate']:.4f}")
        print(f"  Mean CATE:       {metrics['cate_mean']:.4f}")
        print(f"  Neg. Uplift %:   {metrics['negative_uplift_ratio']:.1f}%")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(full_results).T
    comparison_df = comparison_df.sort_values('qini_auc', ascending=False)
    
    print("\n" + "=" * 80)
    print("FULL DATASET COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df[['qini_auc', 'uplift_at_10', 'uplift_at_20', 'uplift_at_30', 
                         'ate', 'cate_mean', 'negative_uplift_ratio']].round(4))
    
    # -----------------------------------------------------------------
    # 2. CROSS-VALIDATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. CROSS-VALIDATION (5-Fold)")
    print("=" * 80)
    
    cv_results_all = {}
    
    for model_name, pred_df in model_predictions.items():
        print(f"\n🔄 Cross-validating {model_name}...")
        cv_results = cross_validate_model(pred_df, data_df, n_splits=N_SPLITS)
        cv_results_all[model_name] = cv_results
        
        # Summary
        print(f"  Qini AUC:    {np.mean(cv_results['qini_auc']):.4f} ± {np.std(cv_results['qini_auc']):.4f}")
        print(f"  Uplift@10:   {np.mean(cv_results['uplift_at_10']):.2f}% ± {np.std(cv_results['uplift_at_10']):.2f}%")
        print(f"  Uplift@20:   {np.mean(cv_results['uplift_at_20']):.2f}% ± {np.std(cv_results['uplift_at_20']):.2f}%")
    
    # CV Summary DataFrame
    cv_summary = {}
    for model_name, cv_data in cv_results_all.items():
        cv_summary[model_name] = {
            'cv_qini_auc_mean': np.mean(cv_data['qini_auc']),
            'cv_qini_auc_std': np.std(cv_data['qini_auc']),
            'cv_uplift_10_mean': np.mean(cv_data['uplift_at_10']),
            'cv_uplift_10_std': np.std(cv_data['uplift_at_10']),
            'cv_uplift_20_mean': np.mean(cv_data['uplift_at_20']),
            'cv_uplift_20_std': np.std(cv_data['uplift_at_20']),
        }
    
    cv_summary_df = pd.DataFrame(cv_summary).T
    cv_summary_df = cv_summary_df.sort_values('cv_qini_auc_mean', ascending=False)
    
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(cv_summary_df.round(4))
    
    # -----------------------------------------------------------------
    # 3. STATISTICAL SIGNIFICANCE TESTS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    model_names = list(cv_results_all.keys())
    significance_results = []
    
    # Her model çifti için test et
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            print(f"\n📊 {model1} vs {model2}:")
            
            # Qini AUC karşılaştırması
            test_result = statistical_significance_test(
                cv_results_all[model1]['qini_auc'],
                cv_results_all[model2]['qini_auc'],
                metric_name='Qini AUC'
            )
            
            if test_result['test'] != 'failed':
                significance_results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': 'Qini AUC',
                    'mean_diff': test_result['mean_difference'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['significant'],
                    'effect_size': test_result['effect_size']
                })
                
                print(f"  Qini AUC difference: {test_result['mean_difference']:+.4f}")
                print(f"  P-value: {test_result['p_value']:.4f}")
                print(f"  Significant: {'YES' if test_result['significant'] else 'NO'}")
                print(f"  Effect size: {test_result['effect_size']} (d={test_result['cohens_d']:.3f})")
    
    if significance_results:
        sig_df = pd.DataFrame(significance_results)
        print("\n" + "=" * 80)
        print("SIGNIFICANCE TEST RESULTS")
        print("=" * 80)
        print(sig_df.to_string(index=False))
    
    # -----------------------------------------------------------------
    # 4. FINAL RANKING
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. FINAL MODEL RANKING")
    print("=" * 80)
    
    # Her metrik için sıralama
    rankings = {
        'qini_auc': comparison_df['qini_auc'].rank(ascending=False),
        'uplift_at_10': comparison_df['uplift_at_10'].rank(ascending=False),
        'uplift_at_20': comparison_df['uplift_at_20'].rank(ascending=False),
        'cv_qini_auc': cv_summary_df['cv_qini_auc_mean'].rank(ascending=False),
    }
    
    ranking_df = pd.DataFrame(rankings)
    ranking_df['average_rank'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('average_rank')
    
    print("\nModel Rankings (lower is better):")
    print(ranking_df.round(2))
    
    best_model = ranking_df.index[0]
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Based on: Average rank across all metrics")
    
    # -----------------------------------------------------------------
    # 5. SAVE RESULTS
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. SAVING RESULTS")
    print("=" * 80)
    
    # Full comparison
    comparison_df.to_csv('results/comprehensive_model_comparison.csv')
    print("✓ Saved: results/comprehensive_model_comparison.csv")
    
    # CV summary
    cv_summary_df.to_csv('results/cv_model_comparison.csv')
    print("✓ Saved: results/cv_model_comparison.csv")
    
    # Rankings
    ranking_df.to_csv('results/model_rankings.csv')
    print("✓ Saved: results/model_rankings.csv")
    
    # Significance tests
    if significance_results:
        sig_df.to_csv('results/model_significance_tests.csv', index=False)
        print("✓ Saved: results/model_significance_tests.csv")
    
    # Combined report
    report = f"""
{'='*80}
COMPREHENSIVE MODEL COMPARISON REPORT
{'='*80}

BEST MODEL: {best_model}

FULL DATASET RESULTS:
{comparison_df[['qini_auc', 'uplift_at_10', 'uplift_at_20', 'ate']].to_string()}

CROSS-VALIDATION RESULTS:
{cv_summary_df.to_string()}

MODEL RANKINGS:
{ranking_df.to_string()}

{'='*80}
"""
    
    with open('exports/comprehensive_comparison_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved: exports/comprehensive_comparison_report.txt")
    
    print("\n✅ Comprehensive comparison complete!")
    
    return comparison_df


if __name__ == '__main__':
    try:
        comparison_df = compare_all_models_comprehensive()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        raise

