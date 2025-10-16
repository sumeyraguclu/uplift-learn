"""
11_train_rlearner.py
X5 RetailHero - R-Learner Training
Residualization-based Learner: Removes confounding through orthogonalization
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


class RLearner:
    """R-Learner: Residualization-based learner"""
    def __init__(self):
        self.model_y = None
        self.model_t = None
        self.model_uplift = None
        self.is_fitted = False
    
    def fit(self, X, y, treatment):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        
        self.model_y = LogisticRegression(random_state=42, max_iter=1000)
        self.model_y.fit(X, y)
        y_pred = self.model_y.predict_proba(X)[:, 1]
        y_residuals = y - y_pred
        
        self.model_t = LogisticRegression(random_state=42, max_iter=1000)
        self.model_t.fit(X, treatment)
        t_pred = self.model_t.predict_proba(X)[:, 1]
        t_residuals = treatment - t_pred
        
        pseudo_target = y_residuals * t_residuals
        self.model_uplift = XGBRegressor(max_depth=4, n_estimators=100, random_state=42, verbosity=0)
        self.model_uplift.fit(X, pseudo_target)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self.model_uplift.predict(X)


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
    print("R-LEARNER TRAINING")
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
    
    print("\nTraining R-Learner...")
    model = RLearner()
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
    output_path = Path('data/model_rlearner.pkl')
    
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
            'model_name': 'R-Learner',
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