"""
10_train_xlearner.py
X5 RetailHero - X-Learner Training
Cross-Fit Learner: Two-stage procedure with stage-specific models
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')


class XLearner:
    """X-Learner: Two-stage cross-fit learner"""
    def __init__(self):
        self.model_trmnt = None
        self.model_ctrl = None
        self.model_stage2 = None
        self.is_fitted = False
    
    def fit(self, X, y, treatment):
        X = np.asarray(X)
        y = np.asarray(y)
        treatment = np.asarray(treatment)
        
        trmnt_mask = treatment == 1
        ctrl_mask = treatment == 0
        
        self.model_ctrl = XGBClassifier(max_depth=3, n_estimators=50, random_state=42, verbosity=0)
        self.model_trmnt = XGBClassifier(max_depth=3, n_estimators=50, random_state=42, verbosity=0)
        
        self.model_ctrl.fit(X[ctrl_mask], y[ctrl_mask])
        self.model_trmnt.fit(X[trmnt_mask], y[trmnt_mask])
        
        mu_0 = self.model_ctrl.predict_proba(X)[:, 1]
        mu_1 = self.model_trmnt.predict_proba(X)[:, 1]
        
        pseudo_y = np.zeros_like(y, dtype=float)
        pseudo_y[trmnt_mask] = y[trmnt_mask].astype(float) - mu_0[trmnt_mask]
        pseudo_y[ctrl_mask] = mu_1[ctrl_mask] - y[ctrl_mask].astype(float)
        
        self.model_stage2 = XGBRegressor(max_depth=3, n_estimators=50, random_state=42, verbosity=0)
        self.model_stage2.fit(X, pseudo_y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self.model_stage2.predict(X)


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
    print("X-LEARNER TRAINING")
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
    
    X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
        X, y, treatment, test_size=0.2, stratify=treatment, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")
    
    print("\nTraining X-Learner...")
    model = XLearner()
    model.fit(X_train_scaled, y_train.values, treatment_train.values)
    print("✓ Model trained")
    
    print("\nEvaluating...")
    uplift = model.predict(X_test_scaled)
    qini = qini_auc_normalized(y_test.values, uplift, treatment_test.values)
    
    sorted_idx = np.argsort(uplift)[::-1]
    y_sorted = y_test.values[sorted_idx]
    t_sorted = treatment_test.values[sorted_idx]
    n = len(y_test)
    
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
    output_path = Path('data/model_xlearner.pkl')
    
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
            'model_name': 'X-Learner',
            'created_at': datetime.now().isoformat(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(feature_cols)
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Saved: {output_path}")
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()