
# ==================== src/optimize.py ====================
"""
Plan optimizasyonu (Greedy ve OR-Tools)

Bu modül Gün 4'te dolduracağız.
"""

import numpy as np
import pandas as pd
from typing import Dict


def greedy_optimizer(
    uplift: np.ndarray,
    margin: float,
    contact_cost: float,
    budget: float
) -> Dict:
    """
    Greedy optimizasyon: En yüksek kârdan başlayarak seç
    
    Sabit maliyette OPTIMAL çözüm!
    
    Parameters
    ----------
    uplift : array-like
        Müşteri uplift skorları
    margin : float
        Müşteri başı marj (TL)
    contact_cost : float
        Temas maliyeti (TL)
    budget : float
        Toplam bütçe (TL)
    
    Returns
    -------
    result : dict
        {
            'selected_indices': Seçilen müşteri indeksleri,
            'total_profit': Beklenen toplam kâr,
            'budget_used': Kullanılan bütçe,
            'roi': ROI
        }
    
    Examples
    --------
    >>> result = greedy_optimizer(uplift, margin=120, contact_cost=1.2, budget=10000)
    >>> print(f"Selected: {len(result['selected_indices'])} customers")
    >>> print(f"ROI: {result['roi']:.2f}x")
    """
    # TODO: Gün 4'te implement et
    raise NotImplementedError("Bu fonksiyon henüz implement edilmedi")
