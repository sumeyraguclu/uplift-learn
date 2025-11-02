# test_imports.py
print("Testing imports...")

from src import (
    TLearner,
    qini_auc_score,
    greedy_optimizer,
    get_config,
    CATECalibrator
)

print("✅ All imports successful!")
print(f"  TLearner: {TLearner}")
print(f"  Config: {get_config()}")