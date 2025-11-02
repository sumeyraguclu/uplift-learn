# ==================== src/optimize.py ====================
"""
Campaign optimization with ROI constraints

Production-grade optimization for uplift-based campaign planning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

# -------------------- Internal helpers -------------------- #
def _to_1d(a) -> np.ndarray:
    """Force array-like to 1D numpy array."""
    return np.asarray(a).reshape(-1)

def _ensure_indices(indices: Optional[np.ndarray], n: int) -> np.ndarray:
    """
    Ensure indices array exists and matches length n.
    If None or wrong-length, fall back to np.arange(n).
    """
    if indices is None:
        return np.arange(n)
    idx = np.asarray(indices).reshape(-1)
    if len(idx) != n:
        return np.arange(n)
    return idx

# -------------------- Optimizers -------------------- #
def greedy_optimizer(
    uplift: np.ndarray,
    margin: float,
    contact_cost: float,
    budget: float,
    indices: Optional[np.ndarray] = None
) -> Dict:
    """Greedy: pick by profit until budget is exhausted (optimal for fixed cost)."""
    uplift = _to_1d(uplift)
    n = len(uplift)
    indices = _ensure_indices(indices, n)

    profit_per_customer = uplift * margin - contact_cost
    sorted_idx = np.argsort(profit_per_customer)[::-1]

    max_contacts = n if contact_cost <= 0 else int(max(0, budget) / contact_cost)
    n_selected = min(max_contacts, n)
    selected_sorted = sorted_idx[:n_selected]

    mask = np.zeros(n, dtype=bool)
    mask[selected_sorted] = True
    selected_uplift = uplift[selected_sorted]

    total_cost = n_selected * max(contact_cost, 0.0)
    expected_revenue = selected_uplift.sum() * margin
    expected_profit = expected_revenue - total_cost
    roi = expected_profit / total_cost if total_cost > 0 else 0.0

    return {
        'selected_indices': indices[selected_sorted],
        'selected_mask': mask,
        'n_selected': int(n_selected),
        'total_cost': float(total_cost),
        'expected_revenue': float(expected_revenue),
        'expected_profit': float(expected_profit),
        'roi': float(roi),
        'roi_pct': float(roi * 100),
        'avg_uplift': float(selected_uplift.mean() if n_selected > 0 else 0.0),
        'budget_used': float(total_cost),
        'budget_remaining': float(max(0.0, budget - total_cost))
    }


def roi_threshold_optimizer(
    uplift: np.ndarray,
    margin: float,
    contact_cost: float,
    min_roi: float = 0.0,
    indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Select all customers with ROI above threshold (no budget constraint).

    Select i if: uplift_i * margin - contact_cost ≥ min_roi * contact_cost
    <=> uplift_i ≥ (contact_cost * (1 + min_roi)) / margin
    """
    uplift = _to_1d(uplift)
    n = len(uplift)
    indices = _ensure_indices(indices, n)

    threshold_uplift = (contact_cost * (1.0 + float(min_roi))) / margin

    # Robust selection via position indices
    mask = (uplift >= threshold_uplift).astype(bool).reshape(-1)
    sel_idx = np.flatnonzero(mask)
    n_selected = int(sel_idx.size)

    selected_uplift = uplift[sel_idx]
    total_cost = n_selected * max(contact_cost, 0.0)
    expected_revenue = selected_uplift.sum() * margin
    expected_profit = expected_revenue - total_cost
    roi = expected_profit / total_cost if total_cost > 0 else 0.0

    return {
        'selected_indices': indices[sel_idx],
        'selected_mask': mask,
        'n_selected': n_selected,
        'total_cost': float(total_cost),
        'expected_revenue': float(expected_revenue),
        'expected_profit': float(expected_profit),
        'roi': float(roi),
        'roi_pct': float(roi * 100),
        'avg_uplift': float(selected_uplift.mean() if n_selected > 0 else 0.0),
        'threshold_uplift': float(threshold_uplift),
        'min_roi': float(min_roi)
    }


def top_k_optimizer(
    uplift: np.ndarray,
    k: Union[int, float],
    margin: float,
    contact_cost: float,
    indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Select top k customers by uplift.
    If k in (0,1), treated as fraction of the population.
    """
    uplift = _to_1d(uplift)
    n = len(uplift)
    indices = _ensure_indices(indices, n)

    if isinstance(k, float) and 0 < k < 1:
        k = int(round(n * k))
    k = max(0, min(int(k), n))

    sorted_idx = np.argsort(uplift)[::-1]
    selected_idx = sorted_idx[:k]

    mask = np.zeros(n, dtype=bool)
    mask[selected_idx] = True

    selected_uplift = uplift[selected_idx]
    total_cost = k * max(contact_cost, 0.0)
    expected_revenue = selected_uplift.sum() * margin
    expected_profit = expected_revenue - total_cost
    roi = expected_profit / total_cost if total_cost > 0 else 0.0

    return {
        'selected_indices': indices[selected_idx],
        'selected_mask': mask,
        'n_selected': int(k),
        'total_cost': float(total_cost),
        'expected_revenue': float(expected_revenue),
        'expected_profit': float(expected_profit),
        'roi': float(roi),
        'roi_pct': float(roi * 100),
        'avg_uplift': float(selected_uplift.mean() if k > 0 else 0.0)
    }


def compare_strategies(
    uplift: np.ndarray,
    margin: float,
    contact_cost: float,
    budget: float,
    k_values: List[Union[int, float]] = [0.1, 0.2, 0.3, 0.5],
    roi_thresholds: List[float] = [0.0, 0.5, 1.0],
    indices: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compare multiple targeting strategies and return a summary table."""
    uplift = _to_1d(uplift)
    n = len(uplift)
    indices = _ensure_indices(indices, n)

    results = []

    # Greedy (budget-constrained)
    greedy_result = greedy_optimizer(uplift, margin, contact_cost, budget, indices)
    results.append({
        'strategy': 'Greedy (Budget)',
        'n_selected': greedy_result['n_selected'],
        'cost': greedy_result['total_cost'],
        'revenue': greedy_result['expected_revenue'],
        'profit': greedy_result['expected_profit'],
        'roi_pct': greedy_result['roi_pct'],
        'avg_uplift': greedy_result['avg_uplift'],
    })

    # Top-k strategies
    for k in k_values:
        topk_result = top_k_optimizer(uplift, k, margin, contact_cost, indices)
        k_label = f"{k*100:.0f}%" if isinstance(k, float) else f"{k}"
        results.append({
            'strategy': f'Top-{k_label}',
            'n_selected': topk_result['n_selected'],
            'cost': topk_result['total_cost'],
            'revenue': topk_result['expected_revenue'],
            'profit': topk_result['expected_profit'],
            'roi_pct': topk_result['roi_pct'],
            'avg_uplift': topk_result['avg_uplift'],
        })

    # ROI threshold strategies
    for roi_thresh in roi_thresholds:
        roi_result = roi_threshold_optimizer(uplift, margin, contact_cost, roi_thresh, indices)
        results.append({
            'strategy': f'ROI≥{roi_thresh*100:.0f}%',
            'n_selected': roi_result['n_selected'],
            'cost': roi_result['total_cost'],
            'revenue': roi_result['expected_revenue'],
            'profit': roi_result['expected_profit'],
            'roi_pct': roi_result['roi_pct'],
            'avg_uplift': roi_result['avg_uplift'],
        })

    return pd.DataFrame(results)


def calculate_campaign_metrics(
    uplift: np.ndarray,
    treatment: np.ndarray,
    margin: float,
    contact_cost: float,
    selected_mask: Optional[np.ndarray] = None
) -> Dict:
    """Calculate comprehensive campaign metrics (projection or post-hoc)."""
    uplift = _to_1d(uplift)
    treatment = _to_1d(treatment)

    if selected_mask is None:
        selected_mask = np.ones(len(uplift), dtype=bool)
    else:
        selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)

    uplift_selected = uplift[selected_mask]
    treatment_selected = treatment[selected_mask]

    n_total = int(selected_mask.sum())
    n_treated = int((treatment_selected == 1).sum())
    n_control = int((treatment_selected == 0).sum())

    expected_uplift = float(uplift_selected.mean() if n_total > 0 else 0.0)
    expected_incremental_conv = float(uplift_selected.sum())

    total_cost = n_total * max(contact_cost, 0.0)
    expected_revenue = expected_incremental_conv * margin
    expected_profit = expected_revenue - total_cost
    roi = expected_profit / total_cost if total_cost > 0 else 0.0

    return {
        'n_total': n_total,
        'n_treated': n_treated,
        'n_control': n_control,
        'treatment_ratio': (n_treated / n_total) if n_total > 0 else 0.0,
        'avg_uplift': expected_uplift,
        'total_expected_uplift': expected_incremental_conv,
        'total_cost': float(total_cost),
        'cost_per_customer': float(contact_cost),
        'expected_revenue': float(expected_revenue),
        'expected_profit': float(expected_profit),
        'roi': float(roi),
        'roi_pct': float(roi * 100),
        'revenue_per_cost': float(expected_revenue / total_cost) if total_cost > 0 else 0.0
    }


def optimize_with_constraints(
    uplift: np.ndarray,
    margin: float,
    contact_cost: float,
    budget: Optional[float] = None,
    min_roi: Optional[float] = None,
    max_customers: Optional[int] = None,
    indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Optimize with multiple constraints in order:
    1) ROI threshold, 2) Budget limit, 3) Customer count limit
    """
    uplift = _to_1d(uplift)
    n = len(uplift)
    indices = _ensure_indices(indices, n)

    # Start with all customers
    mask = np.ones(n, dtype=bool)

    # ROI threshold
    if min_roi is not None:
        threshold_uplift = (contact_cost * (1.0 + float(min_roi))) / margin
        mask &= (uplift >= threshold_uplift)

    # Budget limit
    if budget is not None and contact_cost > 0:
        max_from_budget = int(max(0, budget) / contact_cost)
        if mask.sum() > max_from_budget:
            eligible_idx = np.flatnonzero(mask)
            eligible_uplift = uplift[eligible_idx]
            top_idx = np.argsort(eligible_uplift)[::-1][:max_from_budget]
            sel_idx = eligible_idx[top_idx]

            mask = np.zeros(n, dtype=bool)
            mask[sel_idx] = True

    # Max customers
    if max_customers is not None and mask.sum() > max_customers:
        eligible_idx = np.flatnonzero(mask)
        eligible_uplift = uplift[eligible_idx]
        top_idx = np.argsort(eligible_uplift)[::-1][:max_customers]
        sel_idx = eligible_idx[top_idx]

        mask = np.zeros(n, dtype=bool)
        mask[sel_idx] = True

    sel_idx = np.flatnonzero(mask)
    n_selected = int(sel_idx.size)
    selected_uplift = uplift[sel_idx]

    total_cost = n_selected * max(contact_cost, 0.0)
    expected_revenue = selected_uplift.sum() * margin
    expected_profit = expected_revenue - total_cost
    roi = expected_profit / total_cost if total_cost > 0 else 0.0

    return {
        'selected_indices': indices[sel_idx],
        'selected_mask': mask,
        'n_selected': n_selected,
        'total_cost': float(total_cost),
        'expected_revenue': float(expected_revenue),
        'expected_profit': float(expected_profit),
        'roi': float(roi),
        'roi_pct': float(roi * 100),
        'avg_uplift': float(selected_uplift.mean() if n_selected > 0 else 0.0),
        'constraints_applied': {
            'budget': None if budget is None else float(budget),
            'min_roi': None if min_roi is None else float(min_roi),
            'max_customers': None if max_customers is None else int(max_customers)
        }
    }
