"""
TLearner Usage Examples
Demonstrates how to use the src.model.TLearner class
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from src.model import TLearner


# ======================== EXAMPLE 1: Basic Usage ========================

def example_basic():
    """Basic usage with default parameters"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # True uplift effect
    true_uplift = X[:, 0] * 0.1 + X[:, 1] * 0.05
    
    # Generate outcomes
    p_control = 1 / (1 + np.exp(-X[:, 0]))
    p_treatment = 1 / (1 + np.exp(-(X[:, 0] + true_uplift)))
    
    y = np.where(
        treatment == 1,
        np.random.binomial(1, p_treatment),
        np.random.binomial(1, p_control)
    )
    
    # Train model
    model = TLearner(random_state=42)
    metrics = model.fit(X, y, treatment, test_size=0.2)
    
    print(f"\nTraining Metrics:")
    print(f"  Control AUC: {metrics['auc_0']:.4f}")
    print(f"  Treatment AUC: {metrics['auc_1']:.4f}")
    
    # Predict
    cate = model.predict(X)
    print(f"\nPredicted CATE:")
    print(f"  Mean: {cate.mean():+.4f}")
    print(f"  Std: {cate.std():.4f}")
    
    # Save/Load
    model.save('models/example_model.pkl')
    loaded_model = TLearner.load('models/example_model.pkl')
    print(f"\nModel saved and loaded successfully")
    print(f"Loaded model: {loaded_model}")


# ======================== EXAMPLE 2: Custom Estimators ========================

def example_custom_estimators():
    """Using custom estimators"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Estimators")
    print("=" * 70)
    
    # Generate data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Generate outcome with bounded probability
    p = 0.3 + X['feature_0'].values * 0.05  # Reduced coefficient
    p = np.clip(p, 0, 1)  # Ensure valid probability range
    y = np.random.binomial(1, p, n_samples)
    
    # Custom estimators
    model = TLearner(
        treatment_estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        control_estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        random_state=42
    )
    
    metrics = model.fit(X, y, treatment, verbose=False)
    
    print(f"Training completed with RandomForest estimators")
    print(f"Control AUC: {metrics['auc_0']:.4f}")
    print(f"Treatment AUC: {metrics['auc_1']:.4f}")
    
    # Detailed predictions
    predictions = model.predict_cate(X)
    
    results_df = pd.DataFrame({
        'p_control': predictions['p_control'],
        'p_treatment': predictions['p_treatment'],
        'cate': predictions['cate']
    })
    
    print(f"\nPrediction Summary:")
    print(results_df.describe())


# ======================== EXAMPLE 3: DataFrame Input ========================

def example_dataframe_input():
    """Using pandas DataFrame as input"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: DataFrame Input")
    print("=" * 70)
    
    # Create DataFrame
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'income': np.random.uniform(20000, 100000, n_samples),
        'recency': np.random.randint(1, 100, n_samples),
        'frequency': np.random.randint(1, 50, n_samples),
        'monetary': np.random.uniform(10, 1000, n_samples)
    })
    
    df['treatment'] = np.random.binomial(1, 0.5, n_samples)
    
    # Generate outcome with uplift effect
    uplift_effect = (df['income'] / 50000) * 0.05
    base_prob = 0.2 + (df['frequency'] / 50) * 0.1
    
    df['y'] = np.where(
        df['treatment'] == 1,
        np.random.binomial(1, np.clip(base_prob + uplift_effect, 0, 1)),
        np.random.binomial(1, np.clip(base_prob, 0, 1))
    )
    
    # Train with DataFrame
    X = df[['age', 'income', 'recency', 'frequency', 'monetary']]
    y = df['y']
    treatment = df['treatment']
    
    model = TLearner(random_state=42)
    model.fit(X, y, treatment, verbose=False)
    
    print(f"Model trained with DataFrame input")
    print(f"Features: {model.feature_cols}")
    
    # Predict
    predictions = model.predict_cate(X)
    
    # Add predictions to DataFrame
    df['cate'] = predictions['cate']
    df['p_control'] = predictions['p_control']
    df['p_treatment'] = predictions['p_treatment']
    
    print(f"\nTop 10 customers by CATE:")
    print(df.nlargest(10, 'cate')[['income', 'frequency', 'cate']])


# ======================== EXAMPLE 4: Model Inspection ========================

def example_model_inspection():
    """Inspecting model parameters and state"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Model Inspection")
    print("=" * 70)
    
    # Train model
    np.random.seed(42)
    X = np.random.randn(500, 8)
    treatment = np.random.binomial(1, 0.5, 500)
    y = np.random.binomial(1, 0.3, 500)
    
    model = TLearner(random_state=42)
    model.fit(X, y, treatment, verbose=False)
    
    # Inspect model
    params = model.get_params()
    
    print(f"Model Parameters:")
    print(f"  Random State: {params['random_state']}")
    print(f"  Is Fitted: {params['is_fitted']}")
    print(f"  Number of Features: {params['n_features']}")
    print(f"\nTraining Metrics:")
    for key, value in params['train_metrics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Representation: {model}")
    
    # Feature importance (if using tree-based models)
    if hasattr(model.model_0, 'feature_importances_'):
        print(f"\nControl Model - Top 3 Feature Importances:")
        importances = model.model_0.feature_importances_
        top_indices = np.argsort(importances)[-3:][::-1]
        for idx in top_indices:
            print(f"  Feature {idx}: {importances[idx]:.4f}")


# ======================== RUN ALL EXAMPLES ========================

if __name__ == '__main__':
    example_basic()
    example_custom_estimators()
    example_dataframe_input()
    example_model_inspection()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)