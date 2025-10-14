
# ==================== src/data.py ====================
"""
Veri yükleme ve hazırlama fonksiyonları

Bu modül, Gün 1'de kullandığımız veri işleme kodlarını içerecek.
Şu an boş, Gün 5'te dolduracağız (pipeline).
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_criteo_sample(path: str = "data/criteo_sample.parquet") -> pd.DataFrame:
    """
    Criteo uplift veri setini yükle
    
    Parameters
    ----------
    path : str
        Veri dosyasının yolu
    
    Returns
    -------
    pd.DataFrame
        Yüklenen veri
    
    Examples
    --------
    >>> df = load_criteo_sample()
    >>> print(df.shape)
    (10000, 16)
    """
    # TODO: Gün 5'te implement et
    raise NotImplementedError("Bu fonksiyon henüz implement edilmedi")


def train_test_split_uplift(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uplift modeling için train/test split
    
    ÖNEMLI: Treatment dengesi korunmalı!
    
    Parameters
    ----------
    df : pd.DataFrame
        Tüm veri
    test_size : float
        Test set oranı
    random_state : int
        Random seed
    
    Returns
    -------
    train, test : Tuple[pd.DataFrame, pd.DataFrame]
        Train ve test setleri
    """
    # TODO: Gün 5'te implement et
    raise NotImplementedError("Bu fonksiyon henüz implement edilmedi")
