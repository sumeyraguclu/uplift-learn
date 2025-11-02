"""
9_train_slearner.py
X5 RetailHero - S-Learner Training
Single-Model Learner: Fit one model with treatment as feature
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class SLearner:
    """Single-Model Learner"""
    def __init__(self, model=None):
        self.model = model or XGBClassifier(
            max_depth=4, n_estimators=100, random_state=42, verbosity=0
        )
        self.is_fitted = False
    
    def fit(self, X, y, treatment):
        X = np.asarray(X)
        y = np.asarray(y)
        treatment = np.asarray(treatment).reshape(-1, 1)
        
        X_with_treatment = np.hstack([X, treatment])
        self.model.fit(X_with_treatment, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        X_ctrl = np.hstack([X, np.zeros((n_samples, 1))])
        X_trmnt = np.hstack([X, np.ones((n_samples, 1))])
        
        mu_0 = self.model.predict_proba(X_ctrl)[:, 1]
        mu_1 = self.model.predict_proba(X_trmnt)[:, 1]
        return mu_1 - mu_0


def qini_auc_normalized(y_true, tau, treatment):
    """Calculate normalized Qini AUC"""
    sorted_idx = np.argsort(tau)[::-1]
    
    y_sorted = y_true[sorted_idx]
    treatment_sorted = treatment[sorted_idx]
    
    n = len(y_true)
    treatment_cumsum = np.cumsum(treatment_sorted == 1)
    control_cumsum = np.cumsum(treatment_sorted == 0)
    
    y_treatment_cumsum = np.cumsum((treatment_sorted == 1) * y_sorted)
    y_control_cumsum = np.cumsum((treatment_sorted == 0) * y_sorted)
    
    treatment_rates = np.divide(y_treatment_cumsum, treatment_cumsum, 
                               where=treatment_cumsum > 0, out=np.zeros_like(y_treatment_cumsum, dtype=float))
    control_rates = np.divide(y_control_cumsum, control_cumsum,
                             where=control_cumsum > 0, out=np.zeros_like(y_control_cumsum, dtype=float))
    
    qini_curve = treatment_rates - control_rates
    qini_auc_val = np.trapz(qini_curve) / n
    
    max_possible = min(np.sum(y_sorted[treatment_sorted == 1]) / (treatment_sorted == 1).sum() if (treatment_sorted == 1).sum() > 0 else 0, 1.0)
    
    return qini_auc_val / max_possible if max_possible > 0 else 0.0


def main():
    print("="*70)
    print("S-LEARNER TRAINING")
    print("="*70)
    
    print("\nLoading data...")
    with open('data/x5_rfm_processed.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    df = data_dict['data']
    
    exclude_cols = ['client_id', 'target', 'treatment', 'rfm_segment']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df['target'].copy()
    treatment = df['treatment'].copy()
    
    # Tüm veri ile eğit (T-Learner gibi)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Total samples: {len(X):,}")
    print(f"Features: {len(feature_cols)}")
    
    print("\nTraining S-Learner on full dataset...")
    model = SLearner()
    model.fit(X_scaled, y.values, treatment.values)
    print("✓ Model trained")
    
    print("\nEvaluating on full dataset...")
    uplift = model.predict(X_scaled)
    qini = qini_auc_normalized(y.values, uplift, treatment.values)
    
    sorted_idx = np.argsort(uplift)[::-1]
    y_sorted = y.values[sorted_idx]
    t_sorted = treatment.values[sorted_idx]
    n = len(y)
    
    def calc_uplift_at(pct):
        idx = slice(0, max(1, int(n * pct / 100)))
        y_slice = y_sorted[idx]
        t_slice = t_sorted[idx]
        if (t_slice == 1).sum() > 0 and (t_slice == 0).sum() > 0:
            return y_slice[t_slice == 1].mean() - y_slice[t_slice == 0].mean()
        return 0.0
    
    uplift_at_10 = calc_uplift_at(10)
    uplift_at_20 = calc_uplift_at(20)
    
    print(f"\nResults:")
    print(f"  Qini AUC:   {qini:.4f}")
    print(f"  Uplift@10:  {uplift_at_10:+.4f}")
    print(f"  Uplift@20:  {uplift_at_20:+.4f}")
    print(f"  Mean:       {uplift.mean():+.4f}")
    print(f"  Std:        {uplift.std():.4f}")
    
    print("\nSaving...")
    output_path = Path('data/model_slearner.pkl')
    
    results = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'uplift_predictions': uplift,
        'metrics': {
            'qini_auc': float(qini),
            'uplift@10': float(uplift_at_10),
            'uplift@20': float(uplift_at_20),
            'uplift_mean': float(uplift.mean()),
            'uplift_std': float(uplift.std())
        },
        'metadata': {
            'model_name': 'S-Learner',
            'created_at': datetime.now().isoformat(),
            'total_size': len(X),
            'n_features': len(feature_cols)
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Saved: {output_path}")
    
    # Create predictions CSV
    print("\nCreating predictions CSV...")
    
    # S-Learner için p_control ve p_treatment hesapla
    # S-Learner'da: uplift = p_treatment - p_control
    # Varsayım: p_control = baseline, p_treatment = baseline + uplift
    baseline = y.mean()  # Baseline conversion rate
    p_control = np.full(len(uplift), baseline)  # Sabit baseline
    p_treatment = p_control + uplift  # Baseline + uplift
    
    # Tüm veri client_id'lerini kullan
    all_client_ids = df['client_id'].values
    
    predictions_df = pd.DataFrame({
        'client_id': all_client_ids,
        'p_control': p_control,
        'p_treatment': p_treatment,
        'cate': uplift,
        'cate_pct': uplift * 100
    })
    predictions_df.to_csv('results/slearner_predictions.csv', index=False)
    print("✓ Saved: results/slearner_predictions.csv")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()

