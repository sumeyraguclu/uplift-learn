# 🎯 Migration Guide - Refactoring Progress

## ✅ Step 1: TLearner Module (COMPLETE)
## ✅ Step 2: Metrics Module (COMPLETE)
## ✅ Step 3: Optimization Module (COMPLETE)

---

## ✅ Step 3: Optimization Module (COMPLETE)

### 1. **src/optimize.py - Production Optimization**
Implemented comprehensive campaign optimization functions:

**Core Optimizers:**
- ✅ `greedy_optimizer()` - Budget-constrained optimal selection
- ✅ `roi_threshold_optimizer()` - Select all profitable customers
- ✅ `top_k_optimizer()` - Top k customers by uplift
- ✅ `optimize_with_constraints()` - Multi-constraint optimization
- ✅ `compare_strategies()` - Compare multiple strategies
- ✅ `calculate_campaign_metrics()` - Comprehensive metrics

**Features:**
```python
from src.optimize import (
    greedy_optimizer,
    roi_threshold_optimizer,
    compare_strategies
)

# Greedy optimization (budget-constrained)
result = greedy_optimizer(
    uplift=cate_scores,
    margin=50.0,
    contact_cost=0.5,
    budget=10000
)

# ROI threshold (select all profitable)
result = roi_threshold_optimizer(
    uplift=cate_scores,
    margin=50.0,
    contact_cost=0.5,
    min_roi=0.5  # 50% minimum ROI
)

# Compare all strategies
comparison = compare_strategies(
    uplift=cate_scores,
    margin=50.0,
    contact_cost=0.5,
    budget=10000,
    k_values=[0.1, 0.2, 0.3],
    roi_thresholds=[0.0, 0.5, 1.0]
)
```

**Theory:**
- **Greedy is OPTIMAL** for fixed costs: maximizes Σ(uplift * margin - cost)
- **ROI Threshold**: uplift ≥ (cost * (1 + min_roi)) / margin
- **Budget Constraint**: Select top customers until budget exhausted

---

### 2. **scripts/10_campaign_planning.py - Refactored**
Simplified from 400+ lines to ~250 lines:

**Before:**
- Inline optimization logic
- Repeated calculations
- Hard-coded parameters

**After:**
- Import from `src.optimize`
- Clean strategy comparison
- Maintainable code

**Key Changes:**
```python
# OLD - inline implementation
for strategy in strategies:
    if strategy['method'] == 'cate_top_pct':
        # 20+ lines of selection logic
        ...

# NEW - use centralized module
comparison = compare_strategies(
    uplift, margin, contact_cost, budget,
    k_values=[0.1, 0.2, 0.3],
    roi_thresholds=[0.0, 0.5, 1.0]
)
```

---

### 3. **examples/optimize_usage.py - Usage Examples**
Seven comprehensive examples:
1. Greedy optimization (budget-constrained)
2. ROI threshold (profitability filter)
3. Top-K strategies
4. Strategy comparison
5. Multi-constraint optimization
6. Campaign metrics calculation
7. Real-world e-commerce scenario

---

## 📊 Optimization Strategies

### Greedy (Budget-Constrained)
```
Algorithm: Sort by uplift descending, select until budget exhausted
Optimal: YES (for fixed contact costs)
Use when: You have a hard budget limit
```

### ROI Threshold
```
Algorithm: Select all customers with ROI ≥ threshold
Optimal: YES (for profitability)
Use when: No budget limit, focus on profitability
```

### Top-K
```
Algorithm: Select top k% or top k customers by uplift
Optimal: NO (but simple and intuitive)
Use when: Need fixed campaign size
```

### Multi-Constraint
```
Algorithm: Apply budget + ROI + size constraints sequentially
Optimal: Depends on constraints
Use when: Complex business requirements
```

---

## 🧪 Testing Step 3

### Quick Test
```bash
# Test optimization module
python examples/optimize_usage.py

# Run refactored campaign planning
python scripts/10_campaign_planning.py
```

### Validation
```python
import numpy as np
from src.optimize import greedy_optimizer, roi_threshold_optimizer

# Generate test data
uplift = np.random.beta(2, 5, 1000) * 0.1
margin = 50.0
cost = 0.5
budget = 500.0

# Test greedy
result = greedy_optimizer(uplift, margin, cost, budget)
assert result['total_cost'] <= budget
assert result['roi_pct'] > 0

# Test ROI threshold
result2 = roi_threshold_optimizer(uplift, margin, cost, min_roi=0.0)
assert all(uplift[result2['selected_mask']] * margin >= cost)

print("✅ All tests passed!")
```

---

## 📋 Files Changed in Step 3

### Implemented
- [x] `src/optimize.py` - All optimization functions
- [x] `scripts/10_campaign_planning.py` - Refactored
- [x] `examples/optimize_usage.py` - Examples

### Dependencies
```python
# Required imports in src/optimize.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
```

---

## 🚀 Next Steps

### Step 4: Configuration Module (`src/config.py`)
- [ ] Centralize all constants (margin, cost, budget)
- [ ] Support for config.yaml
- [ ] Environment-based configs
- [ ] Update all scripts to use config

**Files Needed:**
1. Create `src/config.py`
2. Create `config.yaml` (optional)
3. Update `scripts/*.py` to use config

### Step 5: Calibration Module (`src/calibration.py`)
- [ ] Extract from `scripts/12_prepare_cate.py`
- [ ] Isotonic regression wrapper
- [ ] Calibration plotting helpers
- [ ] CATE adjustment functions

**Files Needed:**
1. `scripts/12_prepare_cate.py` (for extraction)

---

## 💡 Usage Patterns

### Pattern 1: Quick Optimization
```python
from src.optimize import greedy_optimizer

# Simple budget-constrained optimization
result = greedy_optimizer(cate, margin=50, contact_cost=0.5, budget=10000)
print(f"Selected: {result['n_selected']} customers")
print(f"ROI: {result['roi_pct']:.1f}%")
```

### Pattern 2: Strategy Comparison
```python
from src.optimize import compare_strategies

# Compare all strategies
comparison = compare_strategies(
    cate, margin=50, contact_cost=0.5, budget=10000
)

# Find best
best = comparison.sort_values('roi_pct', ascending=False).iloc[0]
print(f"Best strategy: {best['strategy']} (ROI: {best['roi_pct']:.1f}%)")
```

### Pattern 3: Profitability Filter
```python
from src.optimize import roi_threshold_optimizer

# Select only profitable customers (break-even or better)
result = roi_threshold_optimizer(cate, margin=50, contact_cost=0.5)
print(f"Profitable customers: {result['n_selected']}")

# Or with higher ROI requirement
result = roi_threshold_optimizer(cate, margin=50, contact_cost=0.5, min_roi=0.5)
print(f"50%+ ROI customers: {result['n_selected']}")
```

### Pattern 4: Multi-Constraint
```python
from src.optimize import optimize_with_constraints

# Complex optimization
result = optimize_with_constraints(
    cate,
    margin=50,
    contact_cost=0.5,
    budget=10000,      # Budget limit
    min_roi=0.3,       # 30% minimum ROI
    max_customers=5000 # Size limit
)
```

---

## ⚠️ Important Notes

1. **Greedy Optimality**: Greedy is OPTIMAL for fixed contact costs. For variable costs, need knapsack solver.

2. **ROI Calculation**: ROI = (Revenue - Cost) / Cost
   - Profit per customer = uplift * margin - contact_cost
   - ROI = profit / contact_cost
   - min_roi=0.0 means break-even (profit ≥ 0)

3. **Uplift Units**: Ensure uplift is in probability units (0-1 scale)

4. **Negative Uplift**: Some customers may have negative CATE (campaign hurts conversion)
   - Greedy automatically excludes them (negative profit)
   - ROI threshold also filters them out

5. **Budget Utilization**: May not use full budget if:
   - Not enough profitable customers
   - Customer count is not a multiple of budget/cost

---

## 📚 References

- **Greedy Algorithms**: Cormen et al. - "Introduction to Algorithms"
- **Knapsack Problem**: For variable costs (future enhancement)
- **ROI Optimization**: Standard marketing analytics
- **Campaign Planning**: Guelman et al. (2015) - "Optimal targeting in advertising"

---

## 🎯 Progress Summary

| Step | Module | Status | Files | Lines Saved |
|------|--------|--------|-------|-------------|
| 1 | TLearner | ✅ | 3 | ~60 |
| 2 | Metrics | ✅ | 3 | ~170 |
| 3 | Optimize | ✅ | 3 | ~150 |
| 4 | Config | 🔄 | - | - |
| 5 | Calibration | 🔄 | - | - |

**Total Progress: 60% Complete** 🎉

---

## 📊 Performance Comparison

### Before Refactoring
```
scripts/10_campaign_planning.py: 400+ lines
- Inline optimization logic
- Hard-coded parameters
- Difficult to test
- Repeated calculations
- 6 strategy implementations
```

### After Refactoring
```
src/optimize.py: ~400 lines (reusable!)
scripts/10_campaign_planning.py: ~250 lines
- Import from src.optimize
- Clean, readable code
- Easy to test
- DRY principle
- Extensible strategies
```

**Net Result:**
- ✅ Same functionality
- ✅ Better maintainability
- ✅ Reusable across projects
- ✅ Comprehensive testing
- ✅ Documentation

---

## 🔄 Migration Path

### For Existing Code Using Old Script

**Option 1: Direct Migration**
```python
# OLD
# Run script manually, parse CSV outputs

# NEW
from src.optimize import compare_strategies
result = compare_strategies(uplift, margin, cost, budget)
```

**Option 2: Backward Compatible**
```python
# Keep old script for now
# But internally use new module
from src.optimize import greedy_optimizer

# Old script calls new module
result = greedy_optimizer(uplift, margin, cost, budget)
```

**Option 3: Gradual**
```python
# Phase 1: Test new module in parallel
# Phase 2: Compare outputs (old vs new)
# Phase 3: Switch to new module
# Phase 4: Deprecate old script
```

---

## 🧩 Integration Examples

### With TLearner (Step 1)
```python
from src.model import TLearner
from src.optimize import greedy_optimizer

# Train model
model = TLearner()
model.fit(X_train, y_train, treatment_train)

# Predict
cate = model.predict(X_test)

# Optimize campaign
result = greedy_optimizer(cate, margin=50, contact_cost=0.5, budget=10000)
print(f"Selected {result['n_selected']} customers, ROI: {result['roi_pct']:.1f}%")
```

### With Metrics (Step 2)
```python
from src.metrics import qini_auc_score
from src.optimize import roi_threshold_optimizer

# Evaluate model
qini = qini_auc_score(y_test, cate_pred, treatment_test)
print(f"Model quality (Qini): {qini:.4f}")

# Optimize if model is good
if qini > 0.05:
    result = roi_threshold_optimizer(cate_pred, margin=50, contact_cost=0.5)
    print(f"Campaign plan: {result['n_selected']} customers")
```

### Full Pipeline
```python
from src.model import TLearner
from src.metrics import evaluate_uplift_model
from src.optimize import compare_strategies

# 1. Train
model = TLearner()
model.fit(X_train, y_train, treatment_train)

# 2. Predict
cate = model.predict(X_test)

# 3. Evaluate
metrics = evaluate_uplift_model(y_test, cate, treatment_test, X_test)
print(f"Qini AUC: {metrics['qini_auc']:.4f}")

# 4. Optimize
comparison = compare_strategies(cate, margin=50, contact_cost=0.5, budget=10000)
best = comparison.sort_values('roi_pct', ascending=False).iloc[0]

# 5. Deploy
print(f"Deploy: {best['strategy']} → ROI: {best['roi_pct']:.1f}%")
```

---

## 🎓 Learning Resources

### Understanding Greedy Algorithms
```python
"""
Why is greedy optimal for fixed costs?

Problem: max Σ(uplift_i * margin - cost) subject to Σcost ≤ budget

Rewrite: max Σ(uplift_i * margin) - n*cost subject to n ≤ budget/cost

Since cost is constant, maximize: Σ(uplift_i * margin)

Solution: Select customers with highest uplift until budget runs out

This is the greedy algorithm → OPTIMAL! ✅
"""
```

### ROI Threshold Intuition
```python
"""
When is a customer profitable?

Profit = uplift * margin - cost
Profit ≥ min_roi * cost  (for desired ROI)

Therefore:
uplift * margin ≥ cost * (1 + min_roi)
uplift ≥ cost * (1 + min_roi) / margin

Example:
margin = $50, cost = $0.50, min_roi = 0.5 (50%)

threshold = 0.50 * 1.5 / 50 = 0.015 (1.5% uplift needed)

Any customer with uplift ≥ 1.5% is profitable!
"""
```

---

## 🐛 Common Issues & Solutions

### Issue 1: All customers selected
```python
# Problem: No budget constraint applied
result = greedy_optimizer(uplift, margin, cost, budget=None)  # ❌

# Solution: Provide budget
result = greedy_optimizer(uplift, margin, cost, budget=10000)  # ✅
```

### Issue 2: Zero customers selected
```python
# Problem: min_roi too high or all uplift is negative
result = roi_threshold_optimizer(uplift, margin, cost, min_roi=10.0)  # 1000% ROI!

# Solution: Lower threshold or check data
print(f"Max uplift: {uplift.max()}")
print(f"Min profitable uplift: {cost*(1+0.5)/margin}")
result = roi_threshold_optimizer(uplift, margin, cost, min_roi=0.0)  # ✅
```

### Issue 3: ROI calculation seems wrong
```python
# Problem: Uplift in wrong units (percentage vs probability)
uplift_pct = np.array([5.0, 10.0, 15.0])  # ❌ Should be 0.05, 0.10, 0.15
result = greedy_optimizer(uplift_pct, margin=50, cost=0.5, budget=100)
# → Unrealistic revenue!

# Solution: Convert to probability
uplift_prob = uplift_pct / 100  # ✅
result = greedy_optimizer(uplift_prob, margin=50, cost=0.5, budget=100)
```

### Issue 4: Need variable costs
```python
# Problem: Contact cost varies by customer
costs = np.array([0.5, 0.8, 1.2, ...])  # Different costs per customer

# Current: greedy_optimizer assumes fixed cost
# Solution: For now, use average cost or implement knapsack solver

# Workaround: Pre-filter by cost efficiency
efficiency = uplift / costs * margin
top_efficient = np.argsort(efficiency)[::-1][:budget//costs.mean()]
```

---

## 🎯 Next Steps Summary

### Step 4: Configuration (Priority: HIGH)
**Why:** Eliminate magic numbers, centralize parameters
**Effort:** 2-3 hours
**Files:** `src/config.py`, `config.yaml`, update scripts

### Step 5: Calibration (Priority: MEDIUM)
**Why:** Improve CATE accuracy
**Effort:** 3-4 hours  
**Files:** `src/calibration.py`, extract from script 12

### Future Enhancements
- [ ] Variable cost knapsack optimizer (OR-Tools)
- [ ] Incremental revenue curves
- [ ] A/B test sample size calculator
- [ ] Multi-objective optimization (profit + customer experience)
- [ ] Real-time optimization API

---

## ✅ Checklist for Step 3 Completion

- [x] Implement `greedy_optimizer()`
- [x] Implement `roi_threshold_optimizer()`
- [x] Implement `top_k_optimizer()`
- [x] Implement `compare_strategies()`
- [x] Implement `calculate_campaign_metrics()`
- [x] Implement `optimize_with_constraints()`
- [x] Refactor `scripts/10_campaign_planning.py`
- [x] Create `examples/optimize_usage.py`
- [x] Update migration guide
- [x] Test all functions
- [x] Document theory and usage
- [x] Provide integration examples

**Step 3: COMPLETE! ✅**

---

Ready for Step 4? Let's centralize configuration! 🚀# 🎯 Migration Guide - Refactoring Progress

## ✅ Step 1: TLearner Module (COMPLETE)

### Files Changed
- [x] `src/model.py` - Production TLearner implementation
- [x] `scripts/5_train_tlearner.py` - Refactored
- [x] `examples/tlearner_usage.py` - Usage examples

### Key Improvements
- Single source of truth for TLearner logic
- Scikit-learn compatible API
- Save/load functionality
- Comprehensive documentation

---

## ✅ Step 2: Metrics Module (COMPLETE)

### 1. **src/metrics.py - Production Metrics**
Implemented comprehensive uplift evaluation metrics:

**Core Metrics:**
- ✅ `qini_auc_score()` - Qini AUC calculation with proper normalization
- ✅ `uplift_at_k()` - Single k value uplift calculation
- ✅ `uplift_at_k_multiple()` - Multiple k values at once
- ✅ `average_treatment_effect()` - ATE with confidence intervals
- ✅ `treatment_balance_check()` - SMD-based balance validation
- ✅ `qini_curve_data()` - Curve data for visualization
- ✅ `evaluate_uplift_model()` - One-stop comprehensive evaluation

**Features:**
```python
from src.metrics import (
    qini_auc_score,
    uplift_at_k,
    uplift_at_k_multiple,
    average_treatment_effect,
    treatment_balance_check,
    evaluate_uplift_model
)

# Quick evaluation
metrics = evaluate_uplift_model(
    y_true, uplift_pred, treatment, 
    X=X_test,
    k_list=[0.1, 0.2, 0.3]
)

# Individual metrics
qini = qini_auc_score(y_true, uplift_pred, treatment)
u30 = uplift_at_k(y_true, uplift_pred, treatment, k=0.3)
ate = average_treatment_effect(y_true, treatment)
balance = treatment_balance_check(X_test, treatment)
```

---

### 2. **scripts/9_evaluate_uplift_metrics.py - Refactored**
Simplified from 350+ lines to ~180 lines:

**Before:**
- All metrics implemented inline
- Duplicated logic
- Hard to test

**After:**
- Single import: `from src.metrics import evaluate_uplift_model`
- Clean visualization functions
- Maintainable code

**Key Changes:**
```python
# OLD - inline implementation
def qini_auc_fast(y_true, cate, treatment):
    # 50+ lines of implementation
    pass

# NEW - use centralized module
from src.metrics import qini_auc_score

qini = qini_auc_score(y_true, cate, treatment)
```

---

### 3. **examples/metrics_usage.py - Usage Examples**
Six comprehensive examples:
1. Basic metrics calculation
2. Multiple k values
3. Qini curve visualization
4. Treatment imbalance detection
5. Comprehensive evaluation
6. Model comparison

---

## 📊 Metric Interpretations

### Qini AUC
```
> 0.15  : EXCELLENT - Model separates high/low uplift very well
0.05-0.15: GOOD - Model has decent ranking ability
0.0-0.05 : FAIR - Model slightly better than random
< 0     : POOR - Worse than random (model is harmful!)
```

### Uplift@k
```
Top 10%: Highest uplift customers - premium targeting
Top 30%: Core responsive segment - standard campaign
Top 50%: Broad campaign - include cautiously
```

### Treatment Balance (SMD)
```
< 0.1  : Good balance - reliable causal inference
0.1-0.2: OK balance - acceptable but watch out
> 0.2  : Poor balance - potential confounding!
```

---

## 🧪 Testing Step 2

### Quick Test
```bash
# Test metrics module
python examples/metrics_usage.py

# Run refactored evaluation
python scripts/9_evaluate_uplift_metrics.py
```

### Validation
```python
# Compare old vs new implementations
import pandas as pd
import numpy as np
from src.metrics import qini_auc_score, uplift_at_k

# Load test data
df = pd.read_csv('results/tlearner_predictions.csv')
# ... load y_true, treatment

# Test Qini
qini_new = qini_auc_score(y_true, df['cate'].values, treatment)
print(f"Qini AUC (new): {qini_new:.4f}")

# Test Uplift@k
u30_new = uplift_at_k(y_true, df['cate'].values, treatment, k=0.3)
print(f"Uplift@30% (new): {u30_new*100:.2f}%")
```

---

## 📋 Files Changed in Step 2

### Implemented
- [x] `src/metrics.py` - All core metrics
- [x] `scripts/9_evaluate_uplift_metrics.py` - Refactored
- [x] `examples/metrics_usage.py` - Examples

### Dependencies
```python
# Required imports in src/metrics.py
import numpy as np
from scipy import stats
from typing import Union, List, Dict
```

---

## 🚀 Next Steps

### Step 3: Optimization Module (`src/optimize.py`)
Now ready to implement:
- [ ] `greedy_optimizer()` - Fixed cost optimization
- [ ] `knapsack_optimizer()` - Variable cost with OR-Tools
- [ ] ROI calculation helpers
- [ ] Budget constraint handling
- [ ] Refactor `scripts/10_campaign_planning.py`
- [ ] Refactor `scripts/13_optimization_engine_meta.py`

**Files Needed:**
1. `src/optimize.py` (current skeleton)
2. `scripts/10_campaign_planning.py` (working implementation)
3. `scripts/13_optimization_engine_meta.py` (ROI optimization)

### Step 4: Configuration Module
- [ ] Centralize all constants
- [ ] Support for config.yaml
- [ ] Environment-based configs

### Step 5: Calibration Module
- [ ] Extract from `scripts/12_prepare_cate.py`
- [ ] Isotonic regression wrapper
- [ ] Calibration plotting helpers

---

## 💡 Usage Patterns

### Pattern 1: Quick Evaluation
```python
from src.metrics import evaluate_uplift_model

# Everything in one call
metrics = evaluate_uplift_model(y, uplift, treatment, X)
print(f"Qini: {metrics['qini_auc']:.4f}")
print(f"ATE: {metrics['ate']['ate']*100:.2f}%")
```

### Pattern 2: Individual Metrics
```python
from src.metrics import qini_auc_score, uplift_at_k

qini = qini_auc_score(y, uplift, treatment)
u10 = uplift_at_k(y, uplift, treatment, k=0.1)
u30 = uplift_at_k(y, uplift, treatment, k=0.3)
```

### Pattern 3: Model Comparison
```python
from src.metrics import qini_auc_score

models = {'TLearner': cate_t, 'SLearner': cate_s}
results = {}

for name, predictions in models.items():
    qini = qini_auc_score(y, predictions, treatment)
    results[name] = qini

best_model = max(results, key=results.get)
print(f"Best model: {best_model}")
```

### Pattern 4: Visualization
```python
from src.metrics import qini_curve_data
import matplotlib.pyplot as plt

curve = qini_curve_data(y, uplift, treatment)
plt.plot(curve['x'], curve['y'])
plt.xlabel('% Targeted')
plt.ylabel('Cumulative Uplift (%)')
plt.show()
```

---

## ⚠️ Important Notes

1. **Scale Consistency**: 
   - `qini_auc_score()` returns normalized score (-1 to 1)
   - `uplift_at_k()` returns fraction (0 to 1)
   - `uplift_at_k_multiple()` returns percentages (0 to 100)

2. **Treatment Encoding**: Must be binary (0/1)

3. **Missing Groups**: Functions handle edge cases (no treatment/control)

4. **Performance**: All functions optimized with numpy operations

5. **Statistical Significance**: ATE includes confidence intervals

---

## 📚 References

- **Qini Curve**: Radcliffe (2007) - "Using control groups to target on predicted lift"
- **Uplift Modeling**: Guelman et al. (2012) - "Random forests for uplift modeling"
- **Treatment Balance**: Austin (2009) - "Balance diagnostics for comparing matched groups"
- **scikit-uplift**: https://github.com/maks-sh/scikit-uplift

---

## 🎯 Progress Summary

| Step | Module | Status | Files | Lines Saved |
|------|--------|--------|-------|-------------|
| 1 | TLearner | ✅ | 3 | ~60 |
| 2 | Metrics | ✅ | 3 | ~170 |
| 3 | Optimize | 🔄 | - | - |
| 4 | Config | 🔄 | - | - |
| 5 | Calibration | 🔄 | - | - |

**Total Progress: 40% Complete** 🎉

## ✅ What Was Done

### 1. **src/model.py - Production-Ready TLearner**
Moved the working TLearner implementation from `scripts/5_train_tlearner.py` into a clean, reusable module:

**Key Features:**
- ✅ Scikit-learn compatible API (`fit`, `predict`)
- ✅ Supports both numpy arrays and pandas DataFrames
- ✅ Flexible estimators (default: XGBoost, but accepts any sklearn-compatible model)
- ✅ Built-in feature scaling with StandardScaler
- ✅ Train/test split with stratification
- ✅ Model persistence (`save` / `load` methods)
- ✅ Comprehensive metrics tracking
- ✅ Detailed docstrings and type hints

**API Methods:**
```python
from src.model import TLearner

# Initialize
model = TLearner(
    treatment_estimator=None,  # Optional, defaults to XGBoost
    control_estimator=None,    # Optional, defaults to XGBoost
    random_state=42
)

# Fit
metrics = model.fit(X, y, treatment, test_size=0.2, verbose=True)
# Returns: {'auc_0': float, 'auc_1': float, ...}

# Predict (simple)
cate = model.predict(X)  # Returns: np.ndarray

# Predict (detailed)
predictions = model.predict_cate(X)
# Returns: {'p_control': array, 'p_treatment': array, 'cate': array}

# Save/Load
model.save('models/my_model.pkl')
loaded = TLearner.load('models/my_model.pkl')

# Inspect
params = model.get_params()
print(model)  # TLearner(random_state=42, n_features=10, status=fitted)
```

---

### 2. **scripts/5_train_tlearner.py - Refactored**
Simplified the training script to use the centralized `src.model.TLearner`:

**Before (180+ lines):**
- TLearner class defined inline
- Duplicated code
- Hard to maintain

**After (~120 lines):**
- Single import: `from src.model import TLearner`
- Cleaner, more maintainable
- Same functionality, better structure

**Changes:**
```python
# OLD
class TLearner:
    # 100+ lines of implementation
    pass

# NEW
from src.model import TLearner

tlearner = TLearner(random_state=42)
train_metrics = tlearner.fit(X, y, treatment)
predictions = tlearner.predict_cate(X)
tlearner.save('models/tlearner_model.pkl')
```

---

### 3. **examples/tlearner_usage.py - Usage Examples**
Created comprehensive examples showing:
- Basic usage with synthetic data
- Custom estimators (RandomForest, etc.)
- DataFrame input handling
- Model inspection and feature importance

---

## 📋 Migration Checklist

### Files Changed
- [x] `src/model.py` - Implemented production TLearner
- [x] `scripts/5_train_tlearner.py` - Refactored to use src.model
- [x] `examples/tlearner_usage.py` - Created usage examples

### Files to Update Next
- [ ] Other scripts using TLearner:
  - `scripts/6_train_slearner.py` (if exists)
  - `scripts/7_train_xlearner.py` (if exists)
  - `scripts/8_train_rlearner.py` (if exists)

---

## 🧪 Testing

### Quick Test
```bash
# Test the new module
python examples/tlearner_usage.py

# Run refactored training script
python scripts/5_train_tlearner.py
```

### Validation Checks
```python
# Load old model (if compatible)
import pickle
with open('models/tlearner_model.pkl', 'rb') as f:
    old_model = pickle.load(f)

# Train new model
from src.model import TLearner
new_model = TLearner(random_state=42)
new_model.fit(X, y, treatment)

# Compare predictions (should be nearly identical)
import numpy as np
old_predictions = old_model['model_1'].predict_proba(X_scaled)[:, 1] - \
                  old_model['model_0'].predict_proba(X_scaled)[:, 1]
new_predictions = new_model.predict(X)

assert np.allclose(old_predictions, new_predictions, rtol=1e-5)
```

---

## 🔄 Backward Compatibility

### Loading Old Models
If you have existing `models/tlearner_model.pkl` files from the old format:

```python
# Old format (dict with keys)
import pickle
with open('models/tlearner_model.pkl', 'rb') as f:
    old_data = pickle.load(f)

# Convert to new TLearner instance
from src.model import TLearner
model = TLearner(random_state=42)
model.model_0 = old_data['model_0']
model.model_1 = old_data['model_1']
model.scaler = old_data['scaler']
model.feature_cols = old_data['feature_cols']
model.is_fitted = True

# Re-save in new format
model.save('models/tlearner_model_v2.pkl')
```

---

## 📊 Benefits

### Code Quality
- **Single Source of Truth**: Model logic in one place
- **DRY Principle**: No duplicated code
- **Testable**: Easy to unit test the model class
- **Maintainable**: Changes only in one file

### Flexibility
- **Swappable Estimators**: Use any sklearn-compatible model
- **Framework Agnostic**: Works with numpy, pandas, or any array-like
- **Production Ready**: Save/load, versioning, metadata tracking

### Performance
- **Same Speed**: No performance degradation
- **Better Memory**: Cleaner object lifecycle
- **Scalability**: Easier to parallelize or distribute

---

## 🚀 Next Steps

### Step 2: Metrics Module (`src/metrics.py`)
- [ ] Implement `qini_auc_score()`
- [ ] Implement `uplift_at_k()`
- [ ] Add visualization helpers
- [ ] Refactor `scripts/9_evaluate_uplift_metrics.py`

### Step 3: Optimization Module (`src/optimize.py`)
- [ ] Implement `greedy_optimizer()`
- [ ] Implement `knapsack_optimizer()` (OR-Tools)
- [ ] Refactor `scripts/10_campaign_planning.py`
- [ ] Refactor `scripts/13_optimization_engine_meta.py`

### Step 4: Configuration (`src/config.py`)
- [ ] Centralize all constants (margin, cost, budget)
- [ ] Add config.yaml support
- [ ] Update all scripts to use config

### Step 5: Calibration Module (`src/calibration.py`)
- [ ] Extract calibration logic from `scripts/12_prepare_cate.py`
- [ ] Create reusable calibration functions

---

## 💡 Usage Tips

### For Development
```python
# Quick iteration
from src.model import TLearner

model = TLearner(random_state=42)
model.fit(X_train, y_train, treatment_train, verbose=True)
cate = model.predict(X_test)
```

### For Production
```python
# With custom estimators and persistence
from xgboost import XGBClassifier
from src.model import TLearner

model = TLearner(
    treatment_estimator=XGBClassifier(max_depth=3, n_estimators=200),
    control_estimator=XGBClassifier(max_depth=3, n_estimators=200),
    random_state=42
)

model.fit(X, y, treatment, test_size=0.2)
model.save(f'models/tlearner_{version}.pkl')

# Later...
loaded = TLearner.load(f'models/tlearner_{version}.pkl')
predictions = loaded.predict_cate(new_data)
```

### For Experimentation
```python
# Try different estimators easily
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.model import TLearner

estimators = {
    'rf': RandomForestClassifier(n_estimators=100),
    'xgb': XGBClassifier(max_depth=5),
    'gb': GradientBoostingClassifier(n_estimators=100)
}

results = {}
for name, estimator in estimators.items():
    model = TLearner(treatment_estimator=estimator, control_estimator=estimator)
    metrics = model.fit(X, y, treatment, verbose=False)
    results[name] = metrics['auc_0'], metrics['auc_1']
```

---

## ⚠️ Important Notes

1. **Feature Scaling**: The TLearner automatically scales features using StandardScaler. Don't scale twice!

2. **Random State**: Always set `random_state` for reproducibility

3. **Model Size**: Saved models include the scaler and both estimators, can be large (10-100MB)

4. **DataFrame Columns**: When using DataFrames, column names are preserved in `model.feature_cols`

5. **Treatment Encoding**: Treatment must be binary (0/1)

---

## 📚 References

- Original Implementation: `scripts/5_train_tlearner.py` (archived)
- Scikit-uplift: https://github.com/maks-sh/scikit-uplift
- T-Learner Paper: Künzel et al. (2019) - "Metalearners for estimating heterogeneous treatment effects"