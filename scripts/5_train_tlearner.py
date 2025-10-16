"""
X5 RetailHero - T-Learner Model Training
Production-Grade Uplift Modeling with scikit-uplift reference

Reference:
- https://github.com/maks-sh/scikit-uplift/blob/master/notebooks/RetailHero_EN.ipynb
- https://github.com/maks-sh/scikit-uplift/blob/master/notebooks/uplift_model_selection_tutorial.ipynb

T-Learner (Two-Model Learner):
- Fit separate models for treatment and control groups
- CATE(X) = P(Y=1|X,T=1) - P(Y=1|X,T=0)
- More interpretable, better for understanding group behavior
- Standard choice in production uplift modeling
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

# ======================== SETUP ========================

Path('logs').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)

class Logger:
    def __init__(self, log_file='logs/5_train_tlearner.log'):
        self.log_file = log_file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Started: {datetime.now()}\n\n")
    
    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

logger = Logger()

# ======================== T-LEARNER ========================

class TLearner:
    """
    Two-Model Learner for heterogeneous treatment effect estimation
    
    Theory:
    - mu_0(X): E[Y|X, T=0]  (control outcome model)
    - mu_1(X): E[Y|X, T=1]  (treatment outcome model)
    - tau(X) = mu_1(X) - mu_0(X)  (conditional average treatment effect)
    
    Advantages:
    - Separate models → capture group-specific patterns
    - Less bias than S-Learner
    - Interpretable (can understand each group separately)
    - Industry standard
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_0 = None
        self.model_1 = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_fitted = False
    
    def _prepare_features(self, df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = {'client_id', 'target', 'treatment', 'rfm_segment',
                          'rfm_score', 'segment', 'r_score', 'f_score', 'm_score'}
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
        return feature_cols
    
    def fit(self, X, y, treatment, test_size=0.2):
        """Fit T-Learner"""
        logger.log("\n" + "="*70)
        logger.log("T-LEARNER TRAINING")
        logger.log("="*70)
        
        self.feature_cols = X.columns.tolist()
        logger.log(f"Features: {len(self.feature_cols)}")
        logger.log(f"Samples: {len(X):,}")
        logger.log(f"Treatment: {(treatment==1).sum():,} | Control: {(treatment==0).sum():,}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
            X_scaled, y, treatment, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=treatment
        )
        
        logger.log(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Split by treatment group
        T_mask_0_train = T_train == 0
        T_mask_1_train = T_train == 1
        
        X_0_train = X_train[T_mask_0_train]
        y_0_train = y_train[T_mask_0_train]
        
        X_1_train = X_train[T_mask_1_train]
        y_1_train = y_train[T_mask_1_train]
        
        logger.log(f"\nFitting control model (n={len(X_0_train):,})...")
        self.model_0 = XGBClassifier(
            max_depth=5, n_estimators=100, learning_rate=0.1,
            subsample=0.8, random_state=self.random_state, verbosity=0
        )
        self.model_0.fit(X_0_train, y_0_train)
        
        logger.log(f"Fitting treatment model (n={len(X_1_train):,})...")
        self.model_1 = XGBClassifier(
            max_depth=5, n_estimators=100, learning_rate=0.1,
            subsample=0.8, random_state=self.random_state, verbosity=0
        )
        self.model_1.fit(X_1_train, y_1_train)
        
        # Evaluate on test set
        y_pred_0 = self.model_0.predict_proba(X_test)[:, 1]
        y_pred_1 = self.model_1.predict_proba(X_test)[:, 1]
        
        T_mask_0_test = T_test == 0
        T_mask_1_test = T_test == 1
        
        auc_0 = roc_auc_score(y_test[T_mask_0_test], y_pred_0[T_mask_0_test])
        auc_1 = roc_auc_score(y_test[T_mask_1_test], y_pred_1[T_mask_1_test])
        
        logger.log(f"\nControl model AUC: {auc_0:.4f}")
        logger.log(f"Treatment model AUC: {auc_1:.4f}")
        
        self.is_fitted = True
        
        # Return test set for metrics calculation
        return {
            'X_test': X_test,
            'y_test': y_test,
            'T_test': T_test,
            'auc_0': auc_0,
            'auc_1': auc_1
        }
    
    def predict_cate(self, X):
        """Predict Conditional Average Treatment Effect"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        p_0 = self.model_0.predict_proba(X_scaled)[:, 1]
        p_1 = self.model_1.predict_proba(X_scaled)[:, 1]
        
        cate = p_1 - p_0
        
        return {
            'p_control': p_0,
            'p_treatment': p_1,
            'cate': cate
        }


# ======================== DATA LOADING ========================

def load_data(pkl_path='data/x5_rfm_processed.pkl'):
    """Load preprocessed RFM data"""
    logger.log("\n" + "="*70)
    logger.log("LOADING DATA")
    logger.log("="*70)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    df = data['data']
    metadata = data['metadata']
    
    logger.log(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
    logger.log(f"Treatment: {dict(df['treatment'].value_counts())}")
    logger.log(f"Target: {dict(df['target'].value_counts())}")
    
    return df, metadata


def prepare_features(df):
    """Prepare X, y, treatment"""
    exclude_cols = {'client_id', 'target', 'treatment', 'rfm_segment',
                   'rfm_score', 'segment', 'r_score', 'f_score', 'm_score'}
    
    feature_cols = [c for c in df.columns 
                   if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    treatment = df['treatment'].copy()
    client_id = df['client_id'].copy()
    
    return X, y, treatment, client_id, feature_cols


# ======================== MAIN ========================

def main():
    logger.log("="*70)
    logger.log("X5 RETAILHERO - T-LEARNER MODEL TRAINING")
    logger.log("="*70)
    
    # Load data
    df, metadata = load_data()
    X, y, treatment, client_id, feature_cols = prepare_features(df)
    
    # Train T-Learner
    logger.log("\n" + "="*70)
    tlearner = TLearner(random_state=42)
    test_results = tlearner.fit(X, y, treatment, test_size=0.2)
    
    # Predict CATE for all samples
    logger.log("\n" + "="*70)
    logger.log("PREDICTING CATE FOR ALL SAMPLES")
    logger.log("="*70)
    
    predictions = tlearner.predict_cate(X)
    
    logger.log(f"Mean CATE: {predictions['cate'].mean():+.4f}")
    logger.log(f"Std CATE: {predictions['cate'].std():.4f}")
    logger.log(f"CATE range: [{predictions['cate'].min():+.4f}, {predictions['cate'].max():+.4f}]")
    
    # Save predictions
    logger.log("\n" + "="*70)
    logger.log("SAVING RESULTS")
    logger.log("="*70)
    
    predictions_df = pd.DataFrame({
        'client_id': client_id.values,
        'p_control': predictions['p_control'],
        'p_treatment': predictions['p_treatment'],
        'cate': predictions['cate'],
        'cate_pct': predictions['cate'] * 100
    })
    
    predictions_df.to_csv('results/tlearner_predictions.csv', index=False)
    logger.log("Saved: results/tlearner_predictions.csv")
    
    # Save model
    model_data = {
        'model_0': tlearner.model_0,
        'model_1': tlearner.model_1,
        'scaler': tlearner.scaler,
        'feature_cols': tlearner.feature_cols,
        'created_at': datetime.now().isoformat(),
        'train_results': {
            'auc_0': test_results['auc_0'],
            'auc_1': test_results['auc_1']
        }
    }
    
    with open('models/tlearner_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    logger.log("Saved: models/tlearner_model.pkl")
    
    logger.log("\n" + "="*70)
    logger.log("COMPLETE!")
    logger.log("="*70)
    logger.log("Next: python scripts/9_evaluate_uplift_metrics.py")
    
    return predictions_df, tlearner

if __name__ == '__main__':
    predictions_df, tlearner = main()