"""
X5 RetailHero - T-Learner Model Training (Refactored)
Production-Grade Uplift Modeling

Now uses the centralized src.model.TLearner class
"""

import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import warnings

# Import from src module
from src.model import TLearner

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
    
    # Train T-Learner using src.model.TLearner
    logger.log("\n" + "="*70)
    logger.log("TRAINING T-LEARNER (using src.model.TLearner)")
    logger.log("="*70)
    
    tlearner = TLearner(random_state=42)
    train_metrics = tlearner.fit(X, y, treatment, test_size=0.2, verbose=True)
    
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
    
    # Save model using the new save method
    tlearner.save('models/tlearner_model.pkl')
    logger.log("Saved: models/tlearner_model.pkl")
    
    logger.log("\n" + "="*70)
    logger.log("COMPLETE!")
    logger.log("="*70)
    logger.log("Next: python scripts/9_evaluate_uplift_metrics.py")
    
    return predictions_df, tlearner


if __name__ == '__main__':
    predictions_df, tlearner = main()